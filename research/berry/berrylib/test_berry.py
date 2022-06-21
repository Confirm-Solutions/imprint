import itertools
import time

import berrylib.binomial as binomial
import berrylib.dirty_bayes as dirty_bayes
import berrylib.fast_inla as fast_inla
import berrylib.quadrature as quadrature
import berrylib.util as util
import jax
import numpy as np
import pyimprint.grid as grid
import pytest
import scipy.stats
from berrylib.imprint import BerryImprintModel
from pyimprint.bound import TypeIErrorBound
from pyimprint.driver import accumulate_process
from pyimprint.model.binomial import SimpleSelection
from scipy.special import logit


def logistic(x):
    return jax.scipy.special.expit(x)


def test_broadcast():
    assert util.broadcast(np.zeros(10), (4, 10, 5, 6), (1,)).shape == (1, 10, 1, 1)
    assert util.broadcast(np.zeros((4, 5)), (3, 4, 6, 5), (1, 3)).shape == (1, 4, 1, 5)


def test_dirty_bayes():
    fi = fast_inla.FastINLA()
    y_i = np.array([[3, 8, 5, 4]])
    n_i = np.full((1, 4), 15)
    db_stats = dirty_bayes.calc_dirty_bayes(
        y_i,
        n_i,
        fi.mu_0,
        np.full((1, 4), logit(0.3)),
        np.full((1, 4), logit(0.2) - logit(0.3)),
        fi.sigma2_rule,
    )
    expected = [0.939209, 0.995332, 0.98075, 0.963809]
    np.testing.assert_allclose(db_stats["exceedance"][0, :], expected, 1e-6)


def test_simpson_rules():
    for n in range(3, 10, 2):
        a = np.random.uniform(0, 1)
        b = np.random.uniform(3, 4)
        x = np.linspace(a, b, n)
        y = np.cos(x)
        Iscipy = scipy.integrate.simpson(y, x)

        qr = util.simpson_rule(n, a=a, b=b)
        np.testing.assert_allclose(qr.pts, x)
        Itest = np.sum(qr.wts * y)
        np.testing.assert_allclose(Itest, Iscipy)


def test_gauss_rule():
    # test gauss
    qr = util.gauss_rule(6, a=-1, b=1)
    f = (qr.pts - 0.5) ** 11 + (qr.pts + 0.2) ** 7
    Itest = np.sum(qr.wts * f)
    exact = -10.2957
    np.testing.assert_allclose(Itest, exact, atol=1e-4)


def test_composite_simpson():
    qr = util.composite_rule(
        util.simpson_rule, (21, 0, 0.4 * np.pi), (13, 0.4 * np.pi, 0.5 * np.pi)
    )
    I2 = np.sum(qr.wts * np.cos(qr.pts))
    np.testing.assert_allclose(I2, np.sin(0.5 * np.pi))


def test_log_gauss_rule():
    a = 1e-8
    b = 1e3
    qr = util.log_gauss_rule(90, a, b)
    alpha = 0.0005
    beta = 0.000005
    f = scipy.stats.invgamma.pdf(qr.pts, alpha, scale=beta)
    exact = scipy.stats.invgamma.cdf(b, alpha, scale=beta) - scipy.stats.invgamma.cdf(
        a, alpha, scale=beta
    )
    est = np.sum(f * qr.wts)
    np.testing.assert_allclose(est, exact, 1e-14)


def test_vectorized_bisection():
    np.random.seed(10)
    y = np.random.rand(10, 11)

    def f(x):
        return y - x**2

    for tol in [1e-3, 1e-6, 1e-10]:
        soln, obj, iters = quadrature.vectorized_bisection(
            f, np.full_like(y, 0), np.full_like(y, 100), tol=tol
        )
        correct = np.sqrt(y)
        np.testing.assert_allclose(soln, correct, np.sqrt(tol))
        np.testing.assert_allclose(obj, 0, atol=tol)


def test_quadrature():
    n_i = np.full(4, 10)
    y_i = np.array([1, 6, 3, 3])
    fi = fast_inla.FastINLA(sigma2_n=10, sigma2_bounds=(1e-1, 1e2))

    p_sigma2_g_y = quadrature.integrate(
        fi, y_i, n_i, integrate_sigma2=False, n_theta=11
    )
    p_sigma2_g_y /= np.sum(p_sigma2_g_y * fi.sigma2_rule.wts)
    expected = [
        2.062674e00,
        1.521337e00,
        9.040576e-01,
        4.066886e-01,
        1.120525e-01,
        1.760016e-02,
        2.095379e-03,
        2.803709e-04,
        5.743632e-05,
        2.217637e-05,
    ]
    np.testing.assert_allclose(p_sigma2_g_y, expected, 1e-5)

    ti_rule = util.simpson_rule(11, -1, 2)
    p_arm = quadrature.integrate(
        fi,
        y_i,
        n_i,
        integrate_sigma2=True,
        n_theta=11,
        fixed_arm_dim=1,
        fixed_arm_values=ti_rule.pts,
    )
    p_arm /= np.sum(p_arm * ti_rule.wts)
    expected = [
        0.005733,
        0.026603,
        0.094835,
        0.249339,
        0.473299,
        0.651237,
        0.670311,
        0.542336,
        0.361369,
        0.204235,
        0.099637,
    ]
    np.testing.assert_allclose(p_arm, expected, 1e-5)


@pytest.mark.parametrize("method", ["jax", "numpy", "cpp"])
def test_inla_properties(method):
    n_i = np.array([[35, 35], [35, 35]])
    y_i = np.array([[4, 7], [7, 4]])
    data = np.stack((y_i, n_i), axis=-1)
    inla_model = fast_inla.FastINLA(n_arms=2)
    sigma2_post, exceedances, theta_max, theta_sigma, _ = inla_model.numpy_inference(
        data
    )

    # INLA inference should be perfectly symmetric in the arm ordering.
    np.testing.assert_allclose(theta_max[0, :, 0], theta_max[1, :, 1])
    np.testing.assert_allclose(theta_max[1, :, 0], theta_max[0, :, 1])
    np.testing.assert_allclose(theta_sigma[0, :, 0], theta_sigma[1, :, 1])
    np.testing.assert_allclose(theta_sigma[1, :, 0], theta_sigma[0, :, 1])
    np.testing.assert_allclose(exceedances[0], np.flip(exceedances[1]))
    np.testing.assert_allclose(sigma2_post[0], sigma2_post[1])

    # INLA sigma2_post should integrate to 1.0
    sigma2_integral = np.sum(sigma2_post * inla_model.sigma2_rule.wts, axis=-1)
    np.testing.assert_allclose(sigma2_integral, 1.0)


@pytest.mark.parametrize("method", ["jax", "numpy", "cpp"])
def test_fast_inla(method, N=10, iterations=1):
    n_i = np.tile(np.array([20, 20, 35, 35]), (N, 1))
    y_i = np.tile(np.array([0, 1, 9, 10], dtype=np.float64), (N, 1))
    data = np.stack((y_i, n_i), axis=-1)
    inla_model = fast_inla.FastINLA()

    runtimes = []
    for i in range(iterations):
        start = time.time()
        out = inla_model.inference(data, method=method)
        end = time.time()
        # Prevent optimizations by asserting against the output.
        assert out[0].sum() > 0
        runtimes.append(end - start)

    if iterations > 1:
        print("fastest runtime", np.min(runtimes))
        print("median runtime", np.median(runtimes))
        print("us per sample", np.median(runtimes) * 1e6 / N)

    sigma2_post, exceedances, theta_max, theta_sigma = out

    np.testing.assert_allclose(
        theta_max[0, 12],
        np.array([-6.04682818, -2.09586893, -0.21474981, -0.07019088]),
        atol=1e-3,
    )
    correct = np.array(
        [
            1.25954474e02,
            4.52520893e02,
            8.66625278e02,
            5.08333300e02,
            1.30365045e02,
            2.20403048e01,
            3.15183578e00,
            5.50967224e-01,
            2.68365061e-01,
            1.23585852e-01,
            1.13330444e-02,
            5.94800210e-04,
            4.01075571e-05,
            4.92782335e-06,
            1.41605356e-06,
        ]
    )
    np.testing.assert_allclose(sigma2_post[0], correct, rtol=1e-3)
    np.testing.assert_allclose(
        exceedances[0], [0.28306264, 0.4077219, 0.99714174, 0.99904684], atol=1e-3
    )


def test_fast_inla_same_results(N=1, iterations=1_000):
    """Ensure the optimized jax output matches the numpy output."""
    n_i = np.tile(np.array([20, 20, 35, 35]), (N, 1))
    inla_model = fast_inla.FastINLA()
    methods = ["jax", "numpy"]
    key = jax.random.PRNGKey(0)
    y_is = jax.random.uniform(key, shape=(iterations, N, 1)) * n_i[None]
    for y_i in y_is:
        # print(y_i)
        outs = {}
        for method in methods:
            outs[method] = inla_model.inference(
                np.stack((y_i, n_i), axis=-1), method=method
            )
        for method1, method2 in itertools.combinations(methods, 2):
            # sigma2_post, exceedances, theta_max, theta_sigma
            outs1 = outs[method1]
            outs2 = outs[method2]
            np.testing.assert_allclose(outs1[0], outs2[0], atol=1e-3, rtol=1e-2)
            np.testing.assert_allclose(outs1[1], outs2[1], atol=1e-3)
            np.testing.assert_allclose(
                logistic(outs1[2]), logistic(outs2[2]), atol=1e-3
            )


def test_py_binomial(n_arms=2, n_theta_1d=16, sim_size=100):
    """
    Test against the Imprint accumulation and bound routines.
    """
    n_arm_samples = 35
    seed = 10
    # getting an exact match is only possible with n_threads = 1 because
    # parallelism in the imprint accumulator leads to a different order of random
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

    fi = fast_inla.FastINLA(n_arms=n_arms)
    b = BerryImprintModel(fi, n_arm_samples, [0.85])
    acc_o = accumulate_process(b, gr, sim_size, seed, n_threads)

    np.random.seed(seed)
    samples = np.random.uniform(size=(sim_size, n_arm_samples, n_arms))

    theta_tiles = grid.theta_tiles(gr)
    nulls = grid.is_null_per_arm(gr)

    accumulator = binomial.binomial_accumulator(fi.rejection_inference)
    typeI_sum, typeI_score = accumulator(theta_tiles, nulls, samples)
    assert np.all(typeI_sum.to_py() == acc_o.typeI_sum()[0])
    np.testing.assert_allclose(
        typeI_score.to_py(), acc_o.score_sum().reshape(n_tiles, n_arms), 1e-4
    )

    corners = grid.collect_corners(gr)
    tile_radii = grid.radii_tiles(gr)
    sim_sizes = grid.sim_sizes_tiles(gr)
    total, d0, d0u, d1w, d1uw, d2uw = binomial.upper_bound(
        theta_tiles,
        tile_radii,
        corners,
        sim_sizes,
        n_arm_samples,
        typeI_sum.to_py(),
        typeI_score.to_py(),
    )

    delta = 0.025
    critvals = np.array([0.99])
    simple_selection_model = SimpleSelection(fi.n_arms, n_arm_samples, 1, critvals)
    simple_selection_model.critical_values([fi.critical_value])

    ub = TypeIErrorBound()
    kbs = simple_selection_model.make_imprint_bound_state(gr)
    ub.create(kbs, acc_o, gr, delta)

    np.testing.assert_allclose(d0, ub.delta_0()[0])
    np.testing.assert_allclose(d0u, ub.delta_0_u()[0])
    np.testing.assert_allclose(d1w, ub.delta_1()[0], rtol=1e-05)
    np.testing.assert_allclose(d1uw, ub.delta_1_u()[0])
    np.testing.assert_allclose(d2uw, ub.delta_2_u()[0])
    np.testing.assert_allclose(total, ub.get()[0])


def test_rejection_table():
    fi = fast_inla.FastINLA(n_arms=2)
    n = 10
    table = binomial.build_rejection_table(2, n, fi.rejection_inference)

    np.random.seed(11)
    for p in np.linspace(0, 1, 11):
        y = scipy.stats.binom.rvs(n, p, size=(1, fi.n_arms))
        data = np.stack((y, np.full_like(y, n)), axis=-1)
        correct_rej = fi.rejection_inference(data)
        lookup_rej = binomial.lookup_rejection(table, y, n)
        np.testing.assert_allclose(correct_rej, lookup_rej)


if __name__ == "__main__":
    # INLA Benchmark
    N = 10000
    it = 10
    print("jax")
    test_fast_inla("jax", N, it)
    print("cpp")
    test_fast_inla("cpp", N, it)
    print("numpy")
    test_fast_inla("numpy", N, it)

    # Binomial Bound Benchmark
    # test_py_binomial = profile(test_py_binomial)
    # test_py_binomial(3, 32, 10)
