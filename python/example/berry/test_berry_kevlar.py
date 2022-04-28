import fast_inla
import numpy as np
import pykevlar
from binomial_accumulate import binomial_accumulator
from pykevlar.driver import accumulate_process
from pykevlar.grid import HyperPlane, make_cartesian_grid_range
from pykevlar.model.binomial import BerryINLA2
from scipy.special import logit


def test_kevlar_and_py_binomial_accumulate():
    n_arms = 2
    n_arm_samples = 35
    seed = 10
    n_theta_1d = 16
    sim_size = 100
    # getting an exact match is only possible with n_threads = 1 because
    # parallelism in the kevlar accumulator leads to a different order of random
    # numbers.
    n_threads = 1

    # define null hypos
    null_hypos = []
    for i in range(n_arms):
        n = np.zeros(n_arms)
        # null is:
        # theta_i <= logit(0.1)
        # the normal should point towards the negative direction. but that also
        # means we need to negate the logit(0.1) offset
        n[i] = -1
        null_hypos.append(HyperPlane(n, -logit(0.1)))
    gr = make_cartesian_grid_range(
        n_theta_1d, np.full(n_arms, -3.5), np.full(n_arms, 1.0), sim_size
    )
    gr.create_tiles(null_hypos)
    gr.prune()
    n_tiles = gr.n_tiles()

    fi = fast_inla.FastINLA(2)

    # Run the C++ INLA Berry inference/rejection via accumulate_process.
    b = BerryINLA2(
        n_arm_samples,
        [0.85],
        np.full(2, fi.thresh_theta),
        fi.sigma2_rule.wts.copy(),
        fi.cov.reshape((-1, 4)).T.copy(),
        fi.neg_precQ.reshape((-1, 4)).T.copy(),
        fi.logprecQdet.copy(),
        fi.log_prior.copy(),
        fi.tol,
        fi.logit_p1,
    )
    out = accumulate_process(b, gr, sim_size, seed, n_threads)

    # Use the mt19937 object exported from C++ so that we can match the C++ random
    # sequence exactly. This is not necessary in the long term but is temporarily
    # useful to ensure that this code is producing identical output to the C++
    # version.
    n_arm_samples = 35
    gen = pykevlar.mt19937(seed)

    # We flip the order of n_arms and n_arm_samples here so the random number
    # generator produces the same sequence of uniforms as are used in the C++ kevlar
    # internals. The Kevlar function operates in column-major/Fortran order. Whereas
    # here, numpy operates in row-major/C ordering b
    samples = np.empty((sim_size, n_arms, n_arm_samples))
    gen.uniform_sample(samples.ravel())
    # after transposing, samples will have shape (sim_size, n_arm_samples, n_arms)
    samples = np.transpose(samples, (0, 2, 1))

    theta = gr.thetas().T.copy()
    # TODO: it'd be nice to add theta_tiles and is_null_per_arm to the GridRange object!
    cum_n_tiles = np.array(gr.cum_n_tiles())
    n_tiles_per_pt = cum_n_tiles[1:] - cum_n_tiles[:-1]
    theta_tiles = np.repeat(theta, n_tiles_per_pt, axis=0)
    is_null_per_arm = np.array(
        [
            [gr.check_null(i, j) for j in range(n_arms)]
            for i in range(theta_tiles.shape[0])
        ]
    )

    accumulator = binomial_accumulator(fi.rejection_inference)
    typeI_sum, typeI_score = accumulator(theta_tiles, is_null_per_arm, samples)
    print(typeI_score.dtype)
    assert np.all(typeI_sum.to_py() == out.typeI_sum()[0])
    np.testing.assert_allclose(
        typeI_score.to_py(), out.score_sum().reshape(n_tiles, 2), 1e-13
    )


if __name__ == "__main__":
    test_kevlar_and_py_binomial_accumulate()
