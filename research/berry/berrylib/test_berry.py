import time

import berrylib.binomial as binomial
import berrylib.dirty_bayes as dirty_bayes
import berrylib.fast_inla as fast_inla
import berrylib.quadrature as quadrature
import berrylib.util as util
import numpy as np
import pykevlar.grid as grid
import pytest
import scipy.stats
from berrylib.kevlar import BerryKevlarModel
from pykevlar.driver import accumulate_process
from scipy.special import logit


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
    inla_model = fast_inla.FastINLA(n_arms=2)
    sigma2_post, exceedances, theta_max, theta_sigma, _ = inla_model.numpy_inference(
        y_i, n_i
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


@pytest.mark.parametrize("method", ["pytorch"])
def test_fast_inla(method, N=10, iterations=1):
    n_i = np.tile(np.array([20, 20, 35, 35]), (N, 1))
    y_i = np.tile(np.array([0, 1, 9, 10], dtype=np.float64), (N, 1))
    inla_model = fast_inla.FastINLA()

    # import torch
    # inla_model = fast_inla.FastINLA(torch_dtype=torch.float32, torch_device='mps')

    runtimes = []
    for i in range(iterations):
        start = time.time()
        out = inla_model.inference(y_i, n_i, method=method)
        end = time.time()
        runtimes.append(end - start)

    if iterations > 1:
        print("fastest runtime", np.min(runtimes))
        print("median runtime", np.median(runtimes))
        print("us per sample", np.median(runtimes) * 1e6 / N)

    sigma2_post, exceedances, theta_max, theta_sigma = out

    correct_theta_max = [
        [-0.65720195, -0.65720081, -0.65719484, -0.65719371],
        [-0.65720546, -0.65720355, -0.65719345, -0.65719153],
        [-0.65721851, -0.65721369, -0.65718827, -0.65718345],
        [-0.65727494, -0.65725755, -0.65716589, -0.65714851],
        [-0.65757963, -0.65749441, -0.65704505, -0.65695985],
        [-0.65958489, -0.65905323, -0.65625057, -0.65571955],
        [-0.67465157, -0.67076447, -0.65032674, -0.64647392],
        [-0.78705011, -0.75796462, -0.60851132, -0.58150985],
        [-1.34091076, -1.17062088, -0.44985596, -0.34985839],
        [-2.47957009, -1.79535068, -0.28588712, -0.14714995],
        [-3.7880899, -2.05159198, -0.23025555, -0.08632545],
        [-5.02345932, -2.09158471, -0.21765884, -0.07316777],
        [-6.04682818, -2.09586893, -0.21474981, -0.07019088],
        [-6.789137, -2.09644905, -0.21401509, -0.06944441],
        [-7.21806318, -2.096633, -0.21382083, -0.06924673],
    ]
    print(theta_max[0])
    np.testing.assert_allclose(
        theta_max[0],
        correct_theta_max,
        rtol=1e-2,
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
        exceedances[0], [0.28306264, 0.4077219, 0.99714174, 0.99904684], rtol=1e-3
    )


def test_py_binomial_accumulate():
    """
    Test against the Kevlar accumulation routines.
    """
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

    accumulator = binomial.binomial_accumulator(fi.rejection_inference)
    typeI_sum, typeI_score = accumulator(theta_tiles, nulls, samples)
    assert np.all(typeI_sum.to_py() == out.typeI_sum()[0])
    np.testing.assert_allclose(
        typeI_score.to_py(), out.score_sum().reshape(n_tiles, 2), 1e-13
    )


def test_rejection_table():
    fi = fast_inla.FastINLA(2)
    n = 10
    table = binomial.build_rejection_table(2, n, fi.rejection_inference)

    np.random.seed(11)
    for p in np.linspace(0, 1, 11):
        y = scipy.stats.binom.rvs(n, p, size=(1, fi.n_arms))
        correct_rej = fi.rejection_inference(y, np.full_like(y, n))
        lookup_rej = binomial.lookup_rejection(table, y, n)
        np.testing.assert_allclose(correct_rej, lookup_rej)


if __name__ == "__main__":
    N = 10000
    it = 4
    print("jax")
    test_fast_inla("jax", N, it)
    print("cpp")
    test_fast_inla("cpp", N, it)
    print("numpy")
    test_fast_inla("numpy", N, it)
