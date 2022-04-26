import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.stats
import util
from jax.config import config
from scipy.special import logit

# This line is critical for enabling 64-bit floats.
config.update("jax_enable_x64", True)


def fast_invert(S_in, d):
    S = np.tile(S_in, (d.shape[0], 1, 1, 1))
    for k in range(d.shape[-1]):
        outer = np.einsum("...i,...j->...ij", S[..., k, :], S[..., :, k])
        offset = d[..., k] / (1 + d[..., k] * S[..., k, k])
        S = S - (offset[..., None, None] * outer)
    return S


class FastINLA:
    def __init__(self, n_arms=4, sigma2_n=15, critical_value=0.85):
        self.n_arms = n_arms
        self.mu_0 = -1.34
        self.mu_sig_sq = 100.0
        self.logit_p1 = logit(0.3)

        # For numpy impl:
        self.sigma2_n = sigma2_n
        self.sigma2_rule = util.log_gauss_rule(self.sigma2_n, 1e-6, 1e3)
        self.arms = np.arange(self.n_arms)
        self.cov = np.full((self.sigma2_n, self.n_arms, self.n_arms), self.mu_sig_sq)
        self.cov[:, self.arms, self.arms] += self.sigma2_rule.pts[:, None]
        self.neg_precQ = -np.linalg.inv(self.cov)
        self.logprecQdet = 0.5 * np.log(np.linalg.det(-self.neg_precQ))
        self.log_prior = scipy.stats.invgamma.logpdf(
            self.sigma2_rule.pts, 0.0005, scale=0.000005
        )
        self.tol = 1e-3
        self.thresh_theta = logit(0.1) - logit(0.3)
        self.critical_value = critical_value

        # For JAX impl:
        self.sigma2_pts_jax = jnp.asarray(self.sigma2_rule.pts)
        self.sigma2_wts_jax = jnp.asarray(self.sigma2_rule.wts)
        self.cov_jax = jnp.asarray(self.cov)
        self.neg_precQ_jax = jnp.asarray(self.neg_precQ)
        self.logprecQdet_jax = jnp.asarray(self.logprecQdet)
        self.log_prior_jax = jnp.asarray(self.log_prior)

        self.jax_opt_vec = jax.jit(
            jax.vmap(
                jax.vmap(
                    jax_opt,
                    in_axes=(None, None, 0, 0, 0, None, None, None),
                    out_axes=(0, 0),
                ),
                in_axes=(0, 0, None, None, None, None, None, None),
                out_axes=(0, 0),
            )
        )

    def rejection_inference(self, y, n, method="jax"):
        _, exceedance, _, _ = self.inference(y, n, method)
        return exceedance > self.critical_value

    def inference(self, y, n, method="jax"):
        fncs = dict(
            numpy=self.numpy_inference, jax=self.jax_inference, cpp=self.cpp_inference
        )
        return fncs[method](y, n)

    def numpy_inference(self, y, n):
        N = y.shape[0]
        # TODO: warm start with DB theta ?
        # Step 1) Compute the mode of p(theta, y, sigma^2) holding y and sigma^2 fixed.
        # This is a simple Newton's method implementation.
        theta_max = np.zeros((N, self.sigma2_n, self.n_arms))
        converged = False
        for i in range(100):
            theta_m0 = theta_max - self.mu_0
            exp_theta_adj = np.exp(theta_max + self.logit_p1)
            C = 1.0 / (exp_theta_adj + 1)
            grad = (
                np.matmul(self.neg_precQ[None], theta_m0[:, :, :, None])[..., 0]
                + y[:, None]
                - (n[:, None] * exp_theta_adj) * C
            )

            diag = n[:, None] * exp_theta_adj * (C**2)
            hess_inv = fast_invert(-self.cov, -diag)
            step = -np.matmul(hess_inv, grad[..., None])[..., 0]
            theta_max += step
            # hess = np.tile(self.neg_precQ, (N, 1, 1, 1))
            # hess[:, :, self.arms, self.arms] -= (
            #     n[:, None] * exp_theta_adj * (C ** 2)
            # )
            # step = -np.linalg.solve(hess, grad)
            # theta_max += step2
            # np.testing.assert_allclose(step, step2, atol=1e-12)

            if np.max(np.linalg.norm(step, axis=-1)) < self.tol:
                converged = True
                break
        assert converged

        # Step 2) Calculate the joint distribution p(theta, y, sigma^2)
        theta_m0 = theta_max - self.mu_0
        theta_adj = theta_max + self.logit_p1
        exp_theta_adj = np.exp(theta_adj)
        logjoint = (
            0.5 * np.einsum("...i,...ij,...j", theta_m0, self.neg_precQ, theta_m0)
            + self.logprecQdet
            + np.sum(
                theta_adj * y[:, None] - n[:, None] * np.log(exp_theta_adj + 1),
                axis=-1,
            )
            + self.log_prior
        )

        # Step 3) Calculate p(sigma^2 | y) = (
        #   p(theta_max, y, sigma^2)
        #   - log(det(-hessian(theta_max, y, sigma^2)))
        # )
        # The last step in the optimization  will be sufficiently small that we
        # shouldn't need to update the hessian that was calculated during the
        # optimization.
        # hess = np.tile(-precQ, (N, 1, 1, 1))
        # hess[:, :, arms, arms] -= (n[:, None] * np.exp(theta_adj) /
        # ((np.exp(theta_adj) + 1) ** 2))
        log_sigma2_post = logjoint + 0.5 * np.log(np.linalg.det(-hess_inv))
        # This can be helpful for avoiding overflow.
        # log_sigma2_post -= np.max(log_sigma2_post, axis=-1)[:, None] - 600
        sigma2_post = np.exp(log_sigma2_post)
        sigma2_post /= np.sum(sigma2_post * self.sigma2_rule.wts, axis=1)[:, None]

        # Step 4) Calculate p(theta_i | y, sigma^2). This a gaussian
        # approximation using the mode found in the previous optimization step.
        theta_sigma = np.sqrt(np.diagonal(-hess_inv, axis1=2, axis2=3))
        theta_mu = theta_max

        # Step 5) Calculate exceedance probabilities. We do this per sigma^2 and
        # then integrate over sigma^2
        exceedances = []
        for i in range(self.n_arms):
            exc_sigma2 = 1.0 - scipy.stats.norm.cdf(
                self.thresh_theta,
                theta_mu[..., i],
                theta_sigma[..., i],
            )
            exc = np.sum(
                exc_sigma2 * sigma2_post * self.sigma2_rule.wts[None, :], axis=1
            )
            exceedances.append(exc)
        return sigma2_post, np.stack(exceedances, axis=-1), theta_max, theta_sigma

    def jax_inference(self, y, n):
        y = jnp.asarray(y)
        n = jnp.asarray(n)
        theta_max, hess_inv = self.jax_opt_vec(
            y,
            n,
            self.cov_jax,
            self.neg_precQ_jax,
            self.sigma2_pts_jax,
            self.logit_p1,
            self.mu_0,
            self.tol,
        )

        sigma2_post, exceedances, theta_sigma = jax_calc_posterior_and_exceedances(
            theta_max,
            y,
            n,
            self.log_prior_jax,
            self.neg_precQ_jax,
            self.logprecQdet_jax,
            hess_inv,
            self.sigma2_wts_jax,
            self.logit_p1,
            self.mu_0,
            self.thresh_theta,
        )

        return sigma2_post, exceedances, theta_max, theta_sigma

    def cpp_inference(self, y, n):
        import cppimport

        ext = cppimport.imp("fast_inla_ext")
        sigma2_post = np.empty((y.shape[0], self.sigma2_n))
        exceedances = np.empty((y.shape[0], self.n_arms))
        theta_max = np.empty((y.shape[0], self.sigma2_n, self.n_arms))
        theta_sigma = np.empty((y.shape[0], self.sigma2_n, self.n_arms))
        ext.inla_inference(
            sigma2_post,
            exceedances,
            theta_max,
            theta_sigma,
            y,
            n,
            self.sigma2_rule.pts,
            self.sigma2_rule.wts,
            self.log_prior,
            self.neg_precQ,
            self.cov,
            self.logprecQdet,
            self.mu_0,
            self.logit_p1,
            self.tol,
            self.thresh_theta,
        )
        return sigma2_post, exceedances, theta_max, theta_sigma


def jax_opt(y, n, cov, neg_precQ, sigma2, logit_p1, mu_0, tol):
    def step(args):
        theta_max, hess_inv, stop = args
        theta_m0 = theta_max - mu_0
        exp_theta_adj = jnp.exp(theta_max + logit_p1)
        C = 1.0 / (exp_theta_adj + 1)
        nCeta = n * C * exp_theta_adj

        grad = neg_precQ.dot(theta_m0) + y - nCeta
        diag = nCeta * C

        hess_inv = jax_fast_invert(-cov, -diag)
        step = -hess_inv.dot(grad)
        go = jnp.sum(step**2) > tol**2
        return theta_max + step, hess_inv, go

    # NOTE: Warm starting was not helpful but I left this code here in case it's
    # useful.
    # When sigma2 is small, the MLE from summing all the trials is a good guess.
    # When sigma2 is large, the individual arm MLE is a good starting guess.
    # theta_max0 = jnp.where(
    #     sigma2 < 1e-3,
    #     jnp.repeat(jax.scipy.special.logit(y.sum()/n.sum()),self.n_arms) - logit_p1,
    #     jax.scipy.special.logit((y + 1e-4) / n) - logit_p1
    # )
    n_arms = y.shape[0]
    theta_max0 = jnp.zeros(n_arms)

    out = jax.lax.while_loop(
        lambda args: args[2], step, (theta_max0, jnp.zeros((n_arms, n_arms)), True)
    )
    theta_max, hess_inv, stop = out
    return theta_max, hess_inv


def jax_fast_invert(S, d):
    """
    Invert a matrix plus a diagonal by iteratively applying the Sherman-Morrison
    formula. If we are computing Binv = (A + d)^-1,
    then the arguments are:
    - S: A^-1
    - d: d
    """
    # NOTE: It's possible to improve performance by about 10% by doing an
    # incomplete inversion here. In the last iteration through the loop, return
    # both S and offset. Then, perform .dot(grad) with those components directly.
    for k in range(d.shape[0]):
        offset = d[k] / (1 + d[k] * S[k, k])
        S = S - (offset * (S[k, None, :] * S[:, None, k]))
    return S


@jax.jit
def jax_calc_posterior_and_exceedances(
    theta_max,
    y,
    n,
    log_prior,
    neg_precQ,
    logprecQdet,
    hess_inv,
    sigma2_wts,
    logit_p1,
    mu_0,
    thresh_theta,
):
    theta_m0 = theta_max - mu_0
    theta_adj = theta_max + logit_p1
    exp_theta_adj = jnp.exp(theta_adj)
    logjoint = (
        0.5 * jnp.einsum("...i,...ij,...j", theta_m0, neg_precQ, theta_m0)
        + logprecQdet
        + jnp.sum(
            theta_adj * y[:, None] - n[:, None] * jnp.log(exp_theta_adj + 1),
            axis=-1,
        )
        + log_prior
    )

    log_sigma2_post = logjoint + 0.5 * jnp.log(jnp.linalg.det(-hess_inv))
    sigma2_post = jnp.exp(log_sigma2_post)
    sigma2_post /= jnp.sum(sigma2_post * sigma2_wts, axis=1)[:, None]

    theta_sigma = jnp.sqrt(jnp.diagonal(-hess_inv, axis1=2, axis2=3))
    exc_sigma2 = 1.0 - jax.scipy.stats.norm.cdf(
        thresh_theta,
        theta_max,
        theta_sigma,
    )
    exceedances = jnp.sum(
        exc_sigma2 * sigma2_post[:, :, None] * sigma2_wts[None, :, None], axis=1
    )
    return sigma2_post, exceedances, theta_sigma
