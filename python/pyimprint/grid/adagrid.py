import copy
import os

import numpy as np
from pyimprint.bound import TypeIErrorBound
from pyimprint.core.grid import AdaGridInternal, GridRange
from pyimprint.driver import accumulate_process


class AdaGrid(AdaGridInternal):
    """
    AdaGrid (adaptive gridding) is a strategy for
    sampling grid-points in a grid in a way that samples
    more near the points of interest (large Type I error),
    and less in the other regions.
    """

    def __init__(self):
        AdaGridInternal.__init__(self)

    # TODO: This is totally incomplete for now.
    # For now, we let the users pass in initial thresholds.
    def init_thresh__(self, model, grid_range, null_hypo, alpha, seed, n_threads):
        """
        Initializes threshold estimates.

        Note: this assumes that we're doing an one-sided upper-tail test
        because we're always taking the maximum of the thresholds as the
        conservative lambda. This loop is to get a reasonable estimate for the
        thresholds. By construction, they always correspond to threshold such
        that at all initial grid points,
        alpha_hat, alpha_minus_hat <= true alpha, true alpha_minus.
        """

        # model.set_grid_range(grid_range, null_hypo)
        # model_state = model.make_state()

        # sim_sizes = grid_range.sim_sizes()
        # it_o = InitThresh(alpha)
        # gen = mt19937()

        # for j in range(grid_range.size()):
        #    gen.seed(seed)
        #    sim_size_j = sim_sizes[j]
        #    it_o.reset(sim_size_j)
        #    for i in range(sim_size_j):
        #        model_state.rng(gen)
        #        model_state.suff_stat()
        #        it_o.update(model_state, j)
        #    it_o.create(model_state, j)

        # thresh = it_o.thresh()
        # alpha_minus = it_o.alpha_minus()

        # i_star = np.argmax(thresh[0,:]) # argmax of thresh

        # self.alpha_target = alpha
        # self.alpha_minus_target = alpha_minus[i_star]
        # self.thr = thresh[0,i_star]
        # self.thr_minus = thresh[1,i_star]
        # self.da_dthr = (self.alpha_target - self.alpha_minus_target) /
        #        (self.thr-self.thr_minus)

        # print('da_dthresh={dd}, alpha_t={at}, alpha_minus_t={amt}'.format(
        #    dd=self.da_dthr, at=self.alpha_target, amt=self.alpha_minus_target))

    def fit_internal__(
        self,
        batcher,
        model,
        null_hypos,
        grid_range,
        thr,
        thr_minus,
        alpha,
        delta,
        N_max,
        base_seed,
        n_threads,
        finalize_thr,
    ):
        """
        Simulates the model for the current grid range.
        Based on the upper bound object,
        it returns finalized points into grid_final,
        and grid_range as the new set of points.

        """

        # set thresholds for model
        model.critical_values(np.array([thr_minus, thr]))

        # attach batcher to current grid range
        batcher.reset(grid_range=grid_range, null_hypos=null_hypos)

        # TODO: THIS ASSUMES EACH BATCH FINISHES ALL SIMS.
        # Later do a SQL query instead of getting yields,
        # which will remove this problem
        # because by the time the driver is finished
        # the updates are all done regardless of how sims were divided up.
        # For now, we'll just create one big InterSum from all InterSums.
        grs = []
        gfs = []
        for gr, sim_size in batcher:
            is_o = accumulate_process(
                model=model,
                grid_range=gr,
                sim_size=sim_size,
                base_seed=base_seed,
                n_threads=n_threads,
            )
            ub = TypeIErrorBound()
            kbs = model.make_imprint_bound_state(gr)
            ub.create(kbs, is_o, gr, delta)

            # extract estimates of alpha, alpha_minus, N_crit
            d0 = ub.delta_0()
            N = gr.sim_sizes()

            i_star = np.argmax(d0[1, :])
            alpha_hat = d0[1, i_star]
            alpha_minus_hat = d0[0, i_star]

            ntcs = np.cumsum(gr.n_tiles())
            N_crit = N[np.where(ntcs > i_star)[0][0]]

            # call internal C++ routine to update grid ranges
            gf = GridRange()
            self.update(
                ub,
                gr,
                gf,
                N_max,
                finalize_thr,
            )

            # append to list
            grs.append(gr)
            gfs.append(gf)

        def copy_gr(gs, og):
            pos = 0
            for g in gs:
                og.thetas()[:, pos : (pos + g.n_gridpts())] = g.thetas()
                og.radii()[:, pos : (pos + g.n_gridpts())] = g.radii()
                og.sim_sizes()[pos : (pos + g.n_gridpts())] = g.sim_sizes()
                pos += g.n_gridpts()

        grid_range = GridRange(
            grid_range.n_params(), np.sum(np.array([gr.n_gridpts() for gr in grs]))
        )
        grid_final = GridRange(
            grid_range.n_params(), np.sum(np.array([gf.n_gridpts() for gf in gfs]))
        )
        copy_gr(grs, grid_range)
        copy_gr(gfs, grid_final)

        return alpha_hat, alpha_minus_hat, N_crit, grid_range, grid_final

    def fit(
        self,
        batcher,
        model,
        null_hypos,
        init_grid,
        alpha,
        delta,
        seed,
        max_iter,
        N_max,
        alpha_minus,
        thr,
        thr_minus,
        finalize_thr=None,
        n_threads=os.cpu_count(),
        rand_iter=True,
        debug=False,
    ):
        """
        Samples grid-points by piloting the given model
        under the given configuration.

        Parameters
        ----------
        batcher     :   grid-range batch object.
        model       :   model object.
        null_hypo   :   functor whose input is unspecified and is model-specific.
                        Must satisfy model.set_grid_range(..., null_hypo).
        null_hypos  :   list of surface objects that define the null-hypothesis region.
        init_grid   :   initial GridRange object.
        alpha       :   desired nominal level of model test.
        delta       :   1-confidence bound for provable upper bound.
        seed        :   seed for RNG internally.
        max_iter    :   max iteration of splitting grid-points.
        N_max       :   max simulation size.
        finalize_thr:   threshold to determine when a gridpoint is finalized.
                        A gridpoint is finalized if
                        its upper bound value is less than finalize_thr.
                        Default is alpha * 1.1.
        n_threads   :   number of threads for simulation.
        rand_iter   :   True if change seed at every iteration.
                        At iteration i, seed will be seed + i.
                        Note that i=0 is the fit to the initial grid
                        to get an estimate of the thresholds.
                        Otherwise, each iteration will use seed as seed.
                        Default is True.
        debug       :   prints debug messages if True.

        TODO: temporary parameters
        alpha_minus :   target for lower nominal level from alpha.
        thr         :   threshold for test associated with level alpha.
        thr_minus   :   threshold for test associated with level alpha_minus.
        """

        if finalize_thr is None:
            finalize_thr = alpha * 1.1

        # create the first grid range
        grid_range = init_grid

        # list of grid ranges for each iteration that were finalized points.
        grid_finals = []

        # TODO: eventually we want to compute these quantities
        # For now, we get them from user.
        # Initialization is just to get good starting estimates
        # of thr and thr_minus.
        alpha = alpha
        alpha_minus = alpha_minus
        thr = thr
        thr_minus = thr_minus

        itr = 0
        while (grid_range.n_gridpts() > 0) and (itr < max_iter):

            if rand_iter:
                # TODO: how do we ensure that the seed change
                # won't correlate the simulations across iterations?
                # Currently, we are assuming that fit_driver
                # passes the base seed and each process creates
                # base seed + thread-id for each thread.
                # So, the following implementation guarantees uncorrelated data.
                # Possible solution: mangle the seed.
                seed += n_threads

            if debug:
                print(
                    "thr={thr}, thr_minus={thr_minus}".format(
                        thr=thr, thr_minus=thr_minus
                    )
                )

            grid_range_old = copy.deepcopy(grid_range)

            # get estimates for alpha_hat, alpha_minus_hat, N_crit, upper bound
            # updates in-place:
            #   - grid_range as the next set of grid-range.
            #   - grid_final is appended with points.
            (
                alpha_hat,
                alpha_minus_hat,
                N_crit,
                grid_range,
                grid_final,
            ) = self.fit_internal__(
                batcher=batcher,
                model=model,
                null_hypos=null_hypos,
                grid_range=grid_range,
                thr=thr,
                thr_minus=thr_minus,
                alpha=alpha,
                delta=delta,
                N_max=N_max,
                base_seed=seed,
                n_threads=n_threads,
                finalize_thr=finalize_thr,
            )

            # append current iteration of final grid-points
            # TODO: eventually, all final points should be stored in SQL.
            grid_finals.append(grid_final)

            if debug:
                print(
                    "alpha={alpha}, alpha_minus={alpha_minus}".format(
                        alpha=alpha_hat, alpha_minus=alpha_minus_hat
                    )
                )

            # update invariants
            alpha_minus = max(
                1e-8,  # just in case the latter becomes too small (or negative)
                alpha - 2 * np.sqrt(alpha * (1.0 - alpha) / N_crit),
            )
            da_dthr = (alpha_hat - alpha_minus_hat) / (thr - thr_minus)
            thr += (alpha - alpha_hat) / da_dthr
            thr_minus += (alpha_minus - alpha_minus_hat) / da_dthr

            # increment iteration idx
            itr += 1

            # yield current set of grid points we would have returned
            # if this were the last iteration.
            yield grid_range_old, grid_finals
