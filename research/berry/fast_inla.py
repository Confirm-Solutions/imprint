import numpy as np
from scipy.special import logit
import scipy.stats
import scipy.linalg
import jax
import jax.numpy as jnp

# This line is critical for enabling 64-bit floats.
from jax.config import config

config.update("jax_enable_x64", True)

import util


@profile
def fast_invert(S_in, d):
    S = np.tile(S_in, (d.shape[0], 1, 1, 1))
    for k in range(d.shape[-1]):
        outer = np.einsum("...i,...j->...ij", S[..., k, :], S[..., :, k])
        offset = d[..., k] / (1 + d[..., k] * S[..., k, k])
        S = S - (offset[..., None, None] * outer)
    return S


class FastINLA:
    def __init__(self, sigma2_n=15):
        self.mu_0 = -1.34
        self.mu_sig_sq = 100.0
        self.logit_p1 = logit(0.3)

        # For numpy impl:
        self.sigma2_n = sigma2_n
        self.sigma2_rule = util.log_gauss_rule(self.sigma2_n, 1e-6, 1e3)
        self.arms = np.arange(4)
        self.cov = np.full((self.sigma2_n, 4, 4), self.mu_sig_sq)
        self.cov[:, self.arms, self.arms] += self.sigma2_rule.pts[:, None]
        self.neg_precQ = -np.linalg.inv(self.cov)
        self.precQdet = np.linalg.det(-self.neg_precQ)
        self.log_prior = scipy.stats.invgamma.logpdf(
            self.sigma2_rule.pts, 0.0005, scale=0.000005
        )
        self.tol = 1e-3
        self.thresh_theta = logit(0.1) - logit(0.3)

        # For JAX impl:
        self.sigma2_pts_jax = jnp.asarray(self.sigma2_rule.pts)
        self.sigma2_wts_jax = jnp.asarray(self.sigma2_rule.wts)
        self.cov_jax = jnp.asarray(self.cov)
        self.neg_precQ_jax = jnp.asarray(self.neg_precQ)
        self.precQdet_jax = jnp.asarray(self.precQdet)
        self.log_prior_jax = jnp.asarray(self.log_prior)

    @profile
    def numpy_inference(self, y, n):
        N = y.shape[0]
        # TODO: warm start with DB theta ?
        # Step 1) Compute the mode of p(theta, y, sigma^2) holding y and sigma^2 fixed.
        # This is a simple Newton's method implementation.
        theta_max = np.zeros((N, self.sigma2_n, 4))
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
        assert(converged)

        # Step 2) Calculate the joint distribution p(theta, y, sigma^2)
        theta_m0 = theta_max - self.mu_0
        theta_adj = theta_max + self.logit_p1
        exp_theta_adj = np.exp(theta_adj)
        logjoint = (
            0.5 * np.einsum("...i,...ij,...j", theta_m0, self.neg_precQ, theta_m0)
            + 0.5 * np.log(self.precQdet)
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
        # hess[:, :, arms, arms] -= n[:, None] * np.exp(theta_adj) / ((np.exp(theta_adj) + 1) ** 2)
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
        for i in range(4):
            exc_sigma2 = 1.0 - scipy.stats.norm.cdf(
                self.thresh_theta,
                theta_mu[..., i],
                theta_sigma[..., i],
            )
            exc = np.sum(
                exc_sigma2 * sigma2_post * self.sigma2_rule.wts[None, :], axis=1
            )
            exceedances.append(exc)
        return sigma2_post, np.stack(exceedances, axis=-1), theta_max

    @profile
    def jax_inference(self, y, n):
        # TODO: jit some of this. super slow right now, but that's not surprising.
        N = y.shape[0]
        y = jnp.asarray(y)
        n = jnp.asarray(n)
        theta_max = jnp.zeros((N, self.sigma2_n, 4), dtype=np.float64)
        for i in range(100):
            theta_m0 = theta_max - self.mu_0
            exp_theta_adj = jnp.exp(theta_max + self.logit_p1)
            C = 1.0 / (exp_theta_adj + 1)
            grad = (
                jnp.matmul(self.neg_precQ_jax[None], theta_m0[:, :, :, None])[..., 0]
                + y[:, None]
                - (n[:, None] * exp_theta_adj) * C
            )

            diag = n[:, None] * exp_theta_adj * (C**2)
            S_in = -self.cov_jax
            d = -diag
            S = jnp.tile(S_in, (d.shape[0], 1, 1, 1))
            for k in range(d.shape[-1]):
                outer = jnp.einsum("...i,...j->...ij", S[..., k, :], S[..., :, k])
                offset = d[..., k] / (1 + d[..., k] * S[..., k, k])
                S = S - (offset[..., None, None] * outer)
            hess_inv = S
            step = -jnp.matmul(hess_inv, grad[..., None])[..., 0]
            theta_max += step
            # hess = np.tile(self.neg_precQ_jax, (N, 1, 1, 1))
            # hess[:, :, self.arms, self.arms] -= (
            #     n[:, None] * exp_theta_adj * (C ** 2)
            # )
            # step = -np.linalg.solve(hess, grad)
            # theta_max += step
            # np.testing.assert_allclose(step, step2, atol=1e-12)

            if np.max(np.linalg.norm(step, axis=-1)) < self.tol:
                break

        theta_m0 = theta_max - self.mu_0
        theta_adj = theta_max + self.logit_p1
        exp_theta_adj = jnp.exp(theta_adj)
        logjoint = (
            0.5 * jnp.einsum("...i,...ij,...j", theta_m0, self.neg_precQ_jax, theta_m0)
            + 0.5 * jnp.log(self.precQdet_jax)
            + jnp.sum(
                theta_adj * y[:, None] - n[:, None] * jnp.log(exp_theta_adj + 1),
                axis=-1,
            )
            + self.log_prior_jax
        )

        # hess_inv = jnp.linalg.inv(hess)
        log_sigma2_post = logjoint + 0.5 * jnp.log(jnp.linalg.det(-hess_inv))
        sigma2_post = jnp.exp(log_sigma2_post)
        sigma2_post /= jnp.sum(sigma2_post * self.sigma2_rule.wts, axis=1)[:, None]

        theta_sigma = jnp.sqrt(jnp.diagonal(-hess_inv, axis1=2, axis2=3))
        theta_mu = theta_max
        exceedances = []
        for i in range(4):
            exc_sigma2 = 1.0 - scipy.stats.norm.cdf(
                self.thresh_theta,
                theta_mu[..., i],
                theta_sigma[..., i],
            )
            exc = np.sum(
                exc_sigma2 * sigma2_post * self.sigma2_rule.wts[None, :], axis=1
            )
            exceedances.append(exc)
        return sigma2_post, np.stack(exceedances, axis=-1), theta_max

    def cpp_inference(self, y, n):
        import cppimport
        ext = cppimport.imp("fast_inla_ext")
        sigma2_post = np.empty((y.shape[0], self.sigma2_n))
        exceedances = np.empty((y.shape[0], 4))
        theta_max = np.empty((y.shape[0], self.sigma2_n, 4))
        print(
            ext.inla_inference(
                sigma2_post,
                exceedances,
                theta_max,
                y,
                n,
                self.sigma2_rule.pts,
                self.sigma2_rule.wts,
                self.log_prior,
                self.neg_precQ,
                self.cov,
                self.precQdet,
                self.mu_0,
                self.logit_p1
            )
        )
        return sigma2_post,exceedances,theta_max
