class Adagrid():
    '''
    Adagrid (adaptive gridding) is a strategy for
    sampling grid-points in a grid in a way that samples
    more near the points of interest (large Type I error),
    and less in the other regions.
    '''

    def __init__(self):
        self.alpha_hat = None       # estimate of alpha
        self.alpha_minus_hat = None # estimate of alpha_minus
        self.thr_ = None            # threshold aiming for alpha
        self.thr_minus_ = None      # threshold aiming for alpha_minus

    def init_thresh__(self,):
        '''
        Initializes threshold estimates.

        Note: this assumes that we're doing an one-sided upper-tail test
        because we're always taking the maximum of the thresholds as the conservative lambda.
        This loop is to get a reasonable estimate for the thresholds.
        By construction, they always correspond to threshold such that
        at all initial grid points,
        alpha_hat, alpha_minus_hat <= true alpha, true alpha_minus.
        '''
        thr_minus = -np.Inf
        thr = -np.Inf
        for pt in grid_q.queue:
            thr_minus_new, thr_new = model.initial_thresh(pt)
            thr_minus = max(thr_minus_new, thr_minus)
            thr = max(thr_new, thr)
        model.da_dthresh = (model.alpha_target - model.alpha_minus_target) / (thr - thr_minus)
        model.thresh = thr
        model.thresh_minus = thr_minus
        print('da_dthresh={dd}, alpha_t={at}, alpha_minus_t={amt}'.format(
            dd=model.da_dthresh, at=model.alpha_target, amt=model.alpha_minus_target))


    def fit(self,
            model,
            null_hypo,
            is_not_alt,
            init_grid,
            alpha,
            delta,
            seed,
            max_iter,
            N_max):
        '''
        Samples grid-points by piloting the given model
        under the given configuration.

        Parameters
        ----------
        model       :   model object.
        null_hypo   :   functor whose input is unspecified and is model-specific.
                        Must satisfy model.set_grid_range(..., null_hypo).
        is_not_alt  :   functor whose input is a grid-point.
                        Returns True if grid-point is not in alterantive space.
        init_grid   :   initial GridRange object.
        alpha       :   desired nominal level of model test.
        delta       :   1-confidence bound for provable upper bound.
        seed        :   seed for RNG internally.
        max_iter    :   max iteration of splitting grid-points.
        N_max       :   max simulation size.
        '''

