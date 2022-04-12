import time

import pytest
import numpy as np
import scipy.stats
from scipy.special import logit

import inla
import fast_inla
import util
import berry
import dirty_bayes
import quadrature


def test_binomial_hierarchical_grad_hess():
    nT = 50
    n_i = np.full((1, 4), nT)
    x_i = np.array([[0.05422, -0.058, 0.5411, 1.1393]])
    # p_i = np.array([[0.5135521136895386, 0.3305150325484877, 0.6320743881220601, 0.7575673322021476]])
    y_i = np.array([[28, 14, 33, 36]]) * nT / 50
    data = np.stack((y_i, n_i), axis=2)
    qv0 = np.array([1.0])
    a = np.array([0.0])
    theta = np.stack((a, qv0), axis=1)
    dx = 0.001
    model = berry.BerryMu()
    model.log_prior = simple_prior

    def calc_numerical_grad(local_x_i, row):
        dx_vec = np.zeros(4)
        dx_vec[row] = dx
        f0 = model.log_joint(model, local_x_i - dx_vec, data, theta)
        f2 = model.log_joint(model, local_x_i + dx_vec, data, theta)
        f0_xonly = model.log_joint_xonly(local_x_i - dx_vec, data, theta)
        f2_xonly = model.log_joint_xonly(local_x_i + dx_vec, data, theta)

        # check that xonly is only dropping terms independent of x
        np.testing.assert_allclose(f2 - f0, f2_xonly - f0_xonly)
        return (f2 - f0) / (2 * dx)

    num_grad = np.empty((1, 4))
    for i in range(4):
        num_grad[:, i] = calc_numerical_grad(x_i, i)
    analytical_grad = model.grad(x_i, data, theta)
    np.testing.assert_allclose(num_grad, analytical_grad, atol=1e-5)

    num_hess = np.empty((1, 4))
    for i in range(4):
        # only calculate the diagonal (i = j)
        dx_vec = np.zeros(4)
        dx_vec[i] = dx
        g0 = calc_numerical_grad(x_i - dx_vec, i)
        g2 = calc_numerical_grad(x_i + dx_vec, i)
        num_hess[:, i] = (g2 - g0) / (2 * dx)
    np.set_printoptions(linewidth=100)
    analytical_hess = model.hess(x_i, data, theta)

    np.testing.assert_allclose(num_hess, analytical_hess, atol=1e-5)


def simple_prior(theta):
    Qv = 1.0 / theta[..., 0]
    return scipy.stats.lognorm.logpdf(Qv, 10.0)


def test_inla_sim(n_sims=100, check=True):
    n_sims = 100
    n_arms = 4
    np.random.seed(100)

    n_patients_per_group = 50

    # The effects are drawn from a distribution with mean 0.5 and variance 1.0
    mean_effect = 0.5
    effect_var = 1.0
    t_i = scipy.stats.norm.rvs(mean_effect, np.sqrt(effect_var), size=(n_sims, n_arms))
    # inverse logit to get probabilities from linear predictors.
    p_i = scipy.special.expit(t_i)
    n_i = np.full_like(p_i, n_patients_per_group)
    # draw actual trial results.
    y_i = scipy.stats.binom.rvs(n_patients_per_group, p_i).reshape(n_i.shape)
    data = np.stack((y_i, n_i), axis=2)

    thresh = np.full((1, 4), -1)

    model = berry.Berry()
    post_theta, logpost_theta_data = inla.calc_posterior_hyper(model, data)
    inla_stats = inla.calc_posterior_x(post_theta, logpost_theta_data, thresh)

    ci025 = inla_stats["mu_appx"] - 1.96 * inla_stats["sigma_appx"]
    ci975 = inla_stats["mu_appx"] + 1.96 * inla_stats["sigma_appx"]
    t_i_offset = t_i - model.logit_p1
    good = (ci025 < t_i_offset) & (t_i_offset < ci975)
    frac_contained = np.sum(good) / (n_sims * n_arms)

    # Set the exact value since the seed should be fixed. I don't know if this
    # will persist across machines, but it works for me for now.
    # It's not a problem that this is lower than 95% because the data generating
    # process is different from the model assumptions.
    if check:
        assert frac_contained == 0.9275

    # Test the optimization!
    # Confirm that x0 is truly the mode/peak of p(x|y,\theta).
    # Check that random shifts of x0 have lower joint density.
    x0 = logpost_theta_data["x0"]
    theta_broadcast = logpost_theta_data["hyper_grid"].reshape((1, -1, 1))
    data_broadcast = data[:, None, :]
    x0f = model.log_joint_xonly(x0, data_broadcast, theta_broadcast)
    for i in range(10):
        x0shift = x0 + np.random.uniform(0, 0.01, size=x0.shape)
        x0shiftf = model.log_joint_xonly(x0shift, data_broadcast, theta_broadcast)
        assert np.all(x0f > x0shiftf)


def test_dirty_bayes():
    b = berry.Berry(sigma2_n=90, sigma2_bounds=(1e-8, 1e3))
    y_i = np.array([[3, 8, 5, 4]])
    n_i = np.full((1, 4), 15)
    print(b.sigma2_rule)
    db_stats = dirty_bayes.calc_dirty_bayes(
        y_i,
        n_i,
        b.mu_0,
        np.full((1, 4), logit(0.3)),
        np.full((1, 4), logit(0.2) - logit(0.3)),
        b.sigma2_rule,
    )
    expected = [0.938858, 0.995331, 0.980741, 0.963675]
    np.testing.assert_allclose(db_stats["exceedance"][0, :], expected, 1e-6)


def test_mu_integration():
    # If we select a sigma2 integration domain that does not include large
    # values, then our mu integration range should be almost complete and the
    # numerical integration should produce results almost exactly equal to the
    # results from the analytical integration. This is useful for testing both!
    b_mu = berry.BerryMu(sigma2_n=90, sigma2_bounds=(1e-8, 1e-3))
    b_no_mu = berry.Berry(sigma2_n=90, sigma2_bounds=(1e-8, 1e-3))

    y_i = np.array([[3, 8, 5, 4]])
    n_i = np.full((1, 4), 15)
    data = np.stack((y_i, n_i), axis=2)

    post_hyper_mu, report_mu = inla.calc_posterior_hyper(b_mu, data)
    post_hyper_no_mu, report_no_mu = inla.calc_posterior_hyper(b_no_mu, data)

    thresh = np.full((1, 4), -0.0)
    mu_stats = inla.calc_posterior_x(post_hyper_mu, report_mu, thresh)
    no_mu_stats = inla.calc_posterior_x(post_hyper_no_mu, report_no_mu, thresh)
    # The match should not be exact!! But it should be close.
    assert np.all(mu_stats["exceedance"] - no_mu_stats["exceedance"] < 0.02)


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
    # plt.plot(np.log(pexp) / np.log(10), f)
    # plt.xlabel('$log_{10}\sigma^2$')
    # plt.ylabel('$PDF$')
    # plt.show()
    # print('exact CDF: ', exact),
    # print('numerical integration CDF: ', est)
    # print('error: ', est - exact)
    np.testing.assert_allclose(est, exact, 1e-14)


def test_exact_integrate():
    n_i = np.full((1, 4), 10)
    y_i = np.array([[1, 6, 3, 3]])
    data = np.stack((y_i, n_i), axis=2)
    b = berry.Berry(sigma2_n=10, sigma2_bounds=(1e-1, 1e2))

    p_sigma2_g_y = exact.integrate(
        b, data, integrate_sigma2=False, integrate_thetas=(0, 1, 2, 3), n_theta=11
    )
    p_sigma2_g_y /= np.sum(p_sigma2_g_y * b.sigma2_rule.wts, axis=1)[:, None]
    expected = [
        7.253775e-01,
        5.361246e-01,
        3.266723e-01,
        1.871160e-01,
        1.390248e-01,
        6.546949e-02,
        6.794999e-03,
        8.639377e-04,
        1.737905e-04,
        6.668053e-05,
    ]
    np.testing.assert_allclose(p_sigma2_g_y[0], expected, 1e-6)


def test_exact_integrate2():
    n_i = np.full((1, 4), 10)
    y_i = np.array([[1, 6, 3, 3]])
    data = np.stack((y_i, n_i), axis=2)
    b = berry.Berry(sigma2_n=90, sigma2_bounds=(1e-1, 1e2))

    p_sigma2_g_y = quadrature.integrate(
        b, data, integrate_sigma2=False, integrate_thetas=(0, 1, 2, 3), n_theta=15
    )
    p_sigma2_g_y /= np.sum(p_sigma2_g_y * b.sigma2_rule.wts, axis=1)[:, None]


# @pytest.mark.parametrize('method', ['jax', 'numpy', 'cpp'])
@pytest.mark.parametrize('method', ['cpp'])
def test_fast_inla(method, N=10, iterations=1):
    n_i = np.tile(np.array([20, 20, 35, 35]), (N, 1))
    y_i = np.tile(np.array([0, 1, 9, 10], dtype=np.float64), (N, 1))
    inla_model = fast_inla.FastINLA()

    runtimes = []
    for i in range(iterations):
        start = time.time()
        if method == "numpy":
            out = inla_model.numpy_inference(y_i, n_i)
        elif method == "jax":
            out = inla_model.jax_inference(y_i, n_i)
        elif method == 'cpp':
            out = inla_model.cpp_inference(y_i, n_i)
        end = time.time()
        runtimes.append(end - start)

    if iterations > 1:
        print("fastest runtime", np.min(runtimes))
        print("median runtime", np.median(runtimes))
        print("us per sample", np.median(runtimes) * 1e6 / N)

    sigma2_post, exceedances, theta_max = out

    np.testing.assert_allclose(
        theta_max[0, 12],
        [-6.04682818, -2.09586893, -0.21474981, -0.07019088],
        rtol=1e-3,
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


if __name__ == "__main__":
    test_fast_inla('jax', 100, 10)
    test_fast_inla('numpy', 100, 10)
    # import time

    # for i in range(5):
    #     start = time.time()
    #     # test_inla_sim(n_sims=1000, check=False)
    #     test_exact_integrate2()
    #     print(time.time() - start)
