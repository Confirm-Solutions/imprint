import inla
import numpy as np
import scipy.stats
import util


def simple_prior(theta):
    a = theta[..., 0]
    Qv = 1.0 / theta[..., 1]
    return scipy.stats.norm.logpdf(a, 0, 1) + scipy.stats.lognorm.logpdf(Qv, 10.0)


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
    model = inla.binomial_hierarchical()
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
    analytical_grad = model.gradx_log_joint(x_i, data, theta)
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
    analytical_hess = model.hessx_log_joint(x_i, data, theta)

    np.testing.assert_allclose(num_hess, analytical_hess, atol=1e-5)


def test_optimizer():
    pass


def test_inla_sim(n_sims=100, check=True):
    n_sims = 100
    n_arms = 4
    np.random.seed(100)

    n_patients_per_group = 50

    # The group effects are drawn from a distribution with mean 0.5 and variance 1.0
    mean_effect = 0.5
    effect_var = 1.0
    t_i = scipy.stats.norm.rvs(mean_effect, np.sqrt(effect_var), size=(n_sims, n_arms))

    # inverse logit to get probabilities from linear predictors.
    p_i = scipy.special.expit(t_i)

    n_i = np.full_like(p_i, n_patients_per_group)

    # draw actual trial results.
    y_i = scipy.stats.binom.rvs(n_patients_per_group, p_i).reshape(n_i.shape)
    data = np.stack((y_i, n_i), axis=2)
    model = inla.binomial_hierarchical()
    model.log_prior = simple_prior

    mu_rule = inla.simpson_rule(13, a=-3, b=1)
    sigma2_rule = inla.simpson_rule(15, a=1e-2, b=1)

    post_theta, logpost_theta_data = inla.calc_posterior_theta(
        model, data, (mu_rule, sigma2_rule)
    )

    thresh = np.full((1, 4), -1)
    inla_stats = inla.calc_posterior_x(post_theta, logpost_theta_data, thresh)

    ci025 = inla_stats["mu_appx"] - 1.96 * inla_stats["sigma_appx"]
    ci975 = inla_stats["mu_appx"] + 1.96 * inla_stats["sigma_appx"]
    good = (ci025 < t_i) & (t_i < ci975)
    frac_contained = np.sum(good) / (n_sims * n_arms)

    # Set the exact value since the seed should be fixed. I don't know if this
    # will persist across machines, but it works for me for now.
    if check:
        assert frac_contained == 0.9425

    # Confirm that x0 is truly the mode/peak of p(x|y,\theta).
    # Check that random shifts of x0 have lower joint density.
    x0 = logpost_theta_data["x0"]
    theta_broadcast = logpost_theta_data["theta_grid"].reshape((1, -1, 2))
    data_broadcast = data[:, None, :]
    x0f = model.log_joint_xonly(x0, data_broadcast, theta_broadcast)
    for i in range(10):
        x0shift = x0 + np.random.uniform(0, 0.01, size=x0.shape)
        x0shiftf = model.log_joint_xonly(x0shift, data_broadcast, theta_broadcast)
        assert np.all(x0f > x0shiftf)


def test_simpson_rules():
    for n in range(3, 10, 2):
        a = np.random.uniform(0, 1)
        b = np.random.uniform(3, 4)
        x = np.linspace(a, b, n)
        y = np.cos(x)
        Iscipy = scipy.integrate.simpson(y, x)

        pts, wts = inla.simpson_rule(n, a=a, b=b)
        np.testing.assert_allclose(pts, x)
        Itest = np.sum(wts * y)
        np.testing.assert_allclose(Itest, Iscipy)


def test_gauss_rule():
    # test gauss
    pts, wts = inla.gauss_rule(6, a=-1, b=1)
    f = (pts - 0.5) ** 11 + (pts + 0.2) ** 7
    Itest = np.sum(wts * f)
    exact = -10.2957
    np.testing.assert_allclose(Itest, exact, atol=1e-4)


def test_composite_simpson():
    pts, wts = inla.composite_rule(
        inla.simpson_rule, (21, 0, 0.4 * np.pi), (13, 0.4 * np.pi, 0.5 * np.pi)
    )
    I2 = np.sum(wts * np.cos(pts))
    np.testing.assert_allclose(I2, np.sin(0.5 * np.pi))


def test_log_gauss_rule():
    a = 1e-8
    b = 1e3
    pexp, wexp = util.log_gauss_rule(90, a, b)
    alpha = 0.0005
    beta = 0.000005
    f = scipy.stats.invgamma.pdf(pexp, alpha, scale=beta)
    exact = scipy.stats.invgamma.cdf(b, alpha, scale=beta) - scipy.stats.invgamma.cdf(
        a, alpha, scale=beta
    )
    est = np.sum(f * wexp)
    # plt.plot(np.log(pexp) / np.log(10), f)
    # plt.xlabel('$log_{10}\sigma^2$')
    # plt.ylabel('$PDF$')
    # plt.show()
    # print('exact CDF: ', exact),
    # print('numerical integration CDF: ', est)
    # print('error: ', est - exact)
    np.testing.assert_allclose(est, exact, 1e-14)


if __name__ == "__main__":
    import time

    for i in range(5):
        start = time.time()
        test_inla_sim(n_sims=1000, check=False)
        print(time.time() - start)
