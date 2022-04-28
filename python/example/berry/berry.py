"""
The Berry model!

y_i ~ Binomial(theta_i + logit(p1), N_i)
theta_i ~ N(mu, sigma2)
mu ~ N(mu_0, S2)
sigma2 ~ InvGamma(a, b)

mu_0 = -2.20
S2 = 100
a = 0.0005
b = 0.000005
(normally p1 = 0.3)
"""

import inla
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import util
from scipy.special import expit, logit


def p_to_theta(p, logit_p1):
    return logit(p) - logit_p1


def theta_to_p(theta, logit_p1):
    return expit(theta + logit_p1)


class Berry(inla.INLAModel):
    def __init__(
        self,
        *,
        n_arms=4,
        p0=None,
        p1=None,
        sigma2_n=90,
        sigma2_bounds=(1e-6, 1e3),
    ):
        """
        sigma2_n_quad: int, the number of quadrature points to use integrating
          over the sigma2 hyperparameter
        sigma2_bounds: a tuple (a, b) specifying the integration limits in the
          sigma2 dimension
        """
        self.n_arms = n_arms
        self.n_stages = 6

        # rate of response below this is the null hypothesis
        self.p0 = p0
        if self.p0 is None:
            self.p0 = np.full(self.n_arms, 0.1)
        # alternative hypothesis!
        self.p1 = p1
        if self.p1 is None:
            self.p1 = np.full(self.n_arms, 0.3)
        self.logit_p1 = logit(self.p1)

        # Interim success criterion:
        # For some of Berry's calculations (e.g. the interim analysis success
        # criterion in Figure 1/2, the midpoint of p0 and p1 is used.)
        # Pr(theta[i] > pmid_theta|data) > pmid_accept
        # or concretely: Pr(theta[i] > 0.2|data) > 0.9
        self.pmid = (self.p0 + self.p1) / 2
        self.pmid_theta = p_to_theta(self.pmid, self.logit_p1)
        self.pmid_accept = 0.9

        # Final evaluation criterion:
        # Accept the alternative hypo if Pr(p[i] > p0|data) > pfinal_thresh[i]
        # Or in terms of theta: Pr(theta[i] > p0_theta|data) > pfinal_thresh[i]
        self.p0_theta = p_to_theta(self.p0, self.logit_p1)
        self.pfinal_thresh = np.full(4, 0.85)

        # Early failure criterion:
        # Pr(theta[i] > pmid_theta|data) < pmid_fail
        self.pmid_fail = 0.05

        # Specify the stopping/success criteria.
        self.suc_thresh = np.empty((self.n_stages, self.n_arms))
        # early stopping condition (check-in #1-5)
        self.suc_thresh[:5] = self.pmid_theta
        # final success criterion (check-in #6)
        self.suc_thresh[5] = self.p0_theta

        # mu ~ N(-1.34, 100)
        self.mu_0 = -1.34
        self.mu_sig_sq = 100.0

        # Quadrature rule over sigma2 from 1e-8 to 1e3 in log space.
        self.sigma2_rule = util.log_gauss_rule(sigma2_n, *sigma2_bounds)
        self.quad_rules = (self.sigma2_rule,)

        # Precompute Q inverse covariance (aka precision) matrices
        na = np.arange(self.n_arms)
        cov = np.full(
            (self.sigma2_rule.pts.shape[0], self.n_arms, self.n_arms), self.mu_sig_sq
        )
        cov[:, na, na] += self.sigma2_rule.pts[:, None]
        self.Q = np.linalg.inv(cov)
        self.Qdet = np.linalg.det(self.Q)

    def log_prior(self, hyper):
        # sigma prior: InvGamma(0.0005, 0.000005)
        sigma2 = hyper[..., 0]
        alpha = 0.0005
        beta = 0.000005
        return scipy.stats.invgamma.logpdf(sigma2, alpha, scale=beta)

    def log_gaussian_x(self, x, hyper, include_det):
        """
        The gaussian latent variables likelihood term of the form:
        x = MVN(mu_0, Sig)

        Parameters
        ----------
        x
            The
        hyper
            The hyperparameter array
        include_det
            Should we include the determinant term?
        """

        Q, Qdet = self.get_Q(hyper[..., 0])
        xmm0 = x - self.mu_0
        out = -0.5 * np.einsum("...i,...ij,...j", xmm0, Q, xmm0)
        if include_det:
            out += 0.5 * np.log(Qdet)
        return out

    def get_Q(self, sigma2):
        unique_s, indices = np.unique(sigma2, return_inverse=True)
        np.testing.assert_allclose(unique_s, self.sigma2_rule.pts)
        return (
            self.Q[indices].reshape((*sigma2.shape, self.n_arms, self.n_arms)),
            self.Qdet[indices].reshape(sigma2.shape),
        )

    def log_binomial(self, x, data):
        y = data[..., 0]
        n = data[..., 1]
        adj_x = x + self.logit_p1
        return np.sum(adj_x * y - n * np.log(np.exp(adj_x) + 1), axis=-1)

    def log_joint(self, x, data, hyper):
        # There are three terms here:
        # 1) The terms from the Gaussian distribution of the latent variables
        #    (indepdent of the data):
        # 2) The term from the response variable (in this case, binomial)
        # 3) The prior on the hyperparameters
        return (
            self.log_gaussian_x(x, hyper, True)
            + self.log_binomial(x, data)
            + self.log_prior(hyper)
        )

    def log_joint_xonly(self, x, data, hyper):
        # See log_joint, we drop the parts not dependent on x.
        term1 = self.log_gaussian_x(x, hyper, False)
        term2 = self.log_binomial(x, data)
        return term1 + term2

    def grad(self, x, data, hyper):
        y = data[..., 0]
        n = data[..., 1]
        Q = self.get_Q(hyper[..., 0])[0]
        xmm0 = x - self.mu_0
        term1 = -np.sum(Q * xmm0[..., None, :], axis=-1)
        adj_x = x + self.logit_p1
        term2 = y - (n * np.exp(adj_x) / (np.exp(adj_x) + 1))
        return term1 + term2

    def hess(self, x, data, hyper):
        n = data[..., 1]
        na = np.arange(self.n_arms)
        H = np.empty((*x.shape, self.n_arms))
        H[:] = -self.get_Q(hyper[..., 0])[0]
        adj_x = x + self.logit_p1
        H[:, :, na, na] -= n * np.exp(adj_x) / ((np.exp(adj_x) + 1) ** 2)
        return H

    def det_neg_hess(self, H):
        return np.linalg.det(-H)

    def sigma2_from_H(self, H):
        na = np.arange(self.n_arms)
        return -np.linalg.inv(H)[..., na, na]

    def newton_step(self, x, data, hyper):
        hess = self.hess(x, data, hyper)
        grad = self.grad(x, data, hyper)
        update = -np.linalg.solve(hess, grad)
        return update


class BerryMu(Berry):
    def __init__(
        self,
        *,
        p0=np.full(4, 0.1),
        p1=np.full(4, 0.3),
        sigma2_n=90,
        sigma2_bounds=(1e-8, 1e3),
    ):
        super().__init__(p0=p0, p1=p1, sigma2_n=sigma2_n, sigma2_bounds=sigma2_bounds)
        self.mu_rule = util.gauss_rule(201, -4, 4)
        self.quad_rules = (self.mu_rule, self.sigma2_rule)

    def log_prior(self, hyper):
        mu = hyper[..., 0]
        # mu prior: N(-2.197, 100)
        mu_prior = scipy.stats.norm.logpdf(mu, self.p0_theta[0], 100)

        # sigma prior: InvGamma(0.0005, 0.000005)
        sigma2 = hyper[..., 1]
        alpha = 0.0005
        beta = 0.000005
        sigma2_prior = scipy.stats.invgamma.logpdf(sigma2, alpha, scale=beta)
        return mu_prior + sigma2_prior

    def log_gaussian_x(self, x, hyper, include_det):
        """
        Gaussian likelihood for the latent variables x for the situation in which the
        precision matrix for those latent variables is diagonal.

        what we are computing is: -x^T Q x / 2 + log(determinant(Q)) / 2

        Sometimes, we may want to leave out the determinant term because it does not
        depend on x. This is useful for computational efficiency when we are
        optimizing an objective function with respect to x.
        """
        n_rows = x.shape[-1]
        mu = hyper[..., 0]
        sigma2 = hyper[..., 1]
        Qv = 1.0 / sigma2
        quadratic = -0.5 * ((x - mu[..., None]) ** 2) * Qv[..., None]
        out = np.sum(quadratic, axis=-1)
        if include_det:
            # determinant of diagonal matrix = prod(diagonal)
            # log(prod(diagonal)) = sum(log(diagonal))
            out += np.log(Qv) * n_rows / 2
        return out

    def grad(self, x, data, hyper):
        y = data[..., 0]
        n = data[..., 1]
        mu = hyper[..., 0]
        Qv = 1.0 / hyper[..., 1]
        term1 = -Qv[..., None] * (x - mu[..., None])
        adj_x = x + self.logit_p1
        term2 = y - (n * np.exp(adj_x) / (np.exp(adj_x) + 1))
        return term1 + term2

    def hess(self, x, data, hyper):
        n = data[..., 1]
        Qv = 1.0 / hyper[..., 1]
        adj_x = x + self.logit_p1
        term1 = -n * np.exp(adj_x) / ((np.exp(adj_x) + 1) ** 2)
        term2 = -Qv[..., None]
        return term1 + term2

    def det_neg_hess(self, H):
        return np.prod(-H, axis=-1)

    def sigma2_from_H(self, H):
        return -(1.0 / H)

    def newton_step(self, x, data, hyper):
        hess = self.hess(x, data, hyper)
        grad = self.grad(x, data, hyper)
        # Diagonal hessian makes the update step simple.
        return -grad / hess


def plot_2d_field(logpost_theta_data, field, levels=None, label=None):
    MM = logpost_theta_data["hyper_grid"][:, :, 0]
    SS = logpost_theta_data["hyper_grid"][:, :, 1]
    log_sigma_grid = np.log10(SS)
    cntf = plt.contourf(MM, log_sigma_grid, field, levels=levels, extend="both")
    plt.contour(
        MM,
        log_sigma_grid,
        field,
        colors="k",
        linestyles="-",
        linewidths=0.5,
        levels=levels,
        extend="both",
    )
    cbar = plt.colorbar(cntf)
    if label is not None:
        cbar.set_label(None)
    plt.xlabel("$\mu$")
    plt.ylabel("$\log_{10} (\sigma^2)$")


def figure1_plot(b, title, data, stats):
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(title)
    outergs = fig.add_gridspec(2, 3, hspace=0.3)
    for i in range(data.shape[0]):
        innergs = outergs[i].subgridspec(
            2, 1, wspace=0, hspace=0, height_ratios=[0.7, 0.3]
        )
        figure1_subplot(innergs[0], innergs[1], i, b, data, stats)

    plt.show()


def figure1_subplot(gridspec0, gridspec1, i, b, data, stats, title=None):
    plt.subplot(gridspec0)

    # expit(mu_map) is the posterior estimate of the mean probability.
    p_post = theta_to_p(stats["theta_map"], b.logit_p1)

    cilow = theta_to_p(stats["cilow"], b.logit_p1)
    cihi = theta_to_p(stats["cihi"], b.logit_p1)

    y = data[:, :, 0]
    n = data[:, :, 1]

    # The simple ratio of success to samples. Binomial "p".
    raw_ratio = y / n

    plt.plot(np.arange(4), raw_ratio[i], "kx")
    plt.plot(np.arange(4), p_post[i], "ko", mfc="none")
    plt.plot(np.arange(4), stats["exceedance"][i], "k ", marker=(8, 2, 0))

    plt.vlines(np.arange(4), cilow[i], cihi[i], color="k", linewidth=1.0)

    if i < 5:
        if title is None:
            title = f"Interim Analysis {i+1}"
        plt.hlines([b.pmid_fail, b.pmid_accept], -1, 4, colors=["k"], linestyles=["--"])
        plt.text(-0.1, 0.91, "Early Success", fontsize=7)
        plt.text(2.4, 0.06, "Early Futility", fontsize=7)
    else:
        if title is None:
            title = "Final Analysis"
        plt.hlines([b.pfinal_thresh[0]], -1, 4, colors=["k"], linestyles=["--"])
        plt.text(-0.1, 0.86, "Final Success", fontsize=7)
    plt.title(title)

    plt.xlim([-0.3, 3.3])
    plt.ylim([0.0, 1.05])
    plt.yticks(np.linspace(0.0, 1.0, 6))
    plt.xlabel("Group")
    plt.ylabel("Probability")

    plt.subplot(gridspec1)
    plt.bar(
        [0, 1, 2, 3],
        n[i],
        tick_label=[str(i) for i in range(4)],
        color=(0.6, 0.6, 0.6, 1.0),
        edgecolor="k",
        zorder=0,
    )
    plt.bar(
        [0, 1, 2, 3],
        y[i],
        color=(0.6, 0.6, 0.6, 1.0),
        hatch="////",
        edgecolor="w",
        lw=1.0,
        zorder=1,
    )
    #         # draw hatch
    # ax1.bar(range(1, 5), range(1, 5), color='none', edgecolor='red',
    # hatch="/", lw=1., zorder = 0)
    # # draw edge
    plt.bar([0, 1, 2, 3], y[i], color="none", edgecolor="k", zorder=2)
    ticks = np.arange(0, 36, 5)
    plt.yticks(ticks, [str(i) if i % 10 == 0 else "" for i in ticks])
    plt.xticks(np.arange(4), ["1", "2", "3", "4"])
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.xlabel("Group")
    plt.ylabel("N")
