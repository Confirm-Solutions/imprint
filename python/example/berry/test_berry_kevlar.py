import fast_inla
import numpy as np
import pykevlar.grid as grid
from binomial import binomial_accumulator
from pykevlar.driver import accumulate_process
from scipy.special import logit

from kevlar import BerryKevlarModel


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
        null_hypos.append(grid.HyperPlane(n, -logit(0.1)))
    gr = grid.make_cartesian_grid_range(
        n_theta_1d, np.full(n_arms, -3.5), np.full(n_arms, 1.0), sim_size
    )
    gr.create_tiles(null_hypos)
    gr.prune()
    n_tiles = gr.n_tiles()

    fi = fast_inla.FastINLA(2)
    b = BerryKevlarModel(fi, n_arm_samples, [0.85])
    out = accumulate_process(b, gr, sim_size, seed, n_threads)

    np.random.seed(seed)
    samples = np.random.uniform(size=(sim_size, n_arm_samples, n_arms))

    theta_tiles = grid.theta_tiles(gr)
    nulls = grid.is_null_per_arm(gr)

    accumulator = binomial_accumulator(fi.rejection_inference)
    typeI_sum, typeI_score = accumulator(theta_tiles, nulls, samples)
    assert np.all(typeI_sum.to_py() == out.typeI_sum()[0])
    np.testing.assert_allclose(
        typeI_score.to_py(), out.score_sum().reshape(n_tiles, 2), 1e-13
    )


if __name__ == "__main__":
    test_kevlar_and_py_binomial_accumulate()
