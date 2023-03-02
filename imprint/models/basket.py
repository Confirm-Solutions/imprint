from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config

# This line is critical for enabling 64-bit floats.

config.update("jax_enable_x64", True)


class BayesianBasket:
    def __init__(self, seed, K, *, n_arm_samples=35):
        self.n_arm_samples = n_arm_samples
        np.random.seed(seed)
        self.samples = np.random.uniform(size=(K, n_arm_samples, 3))
        self.fi = FastINLA(n_arms=3, critical_value=0.95)
        self.family = "binomial"
        self.family_params = {"n": n_arm_samples}

    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        # 1. Calculate the binomial count data.
        # The sufficient statistic for binomial is just the number of uniform draws
        # above the threshold probability. But the `p_tiles` array has shape (n_tiles,
        # n_arms). So, we add empty dimensions to broadcast and then sum across
        # n_arm_samples to produce an output `y` array of shape: (n_tiles,
        # sim_size, n_arms)

        p = jax.scipy.special.expit(theta)
        y = jnp.sum(self.samples[None, begin_sim:end_sim] < p[:, None, None], axis=2)

        # 2. Determine if we rejected each simulated sample.
        # rejection_fnc expects inputs of shape (n, n_arms) so we must flatten
        # our 3D arrays. We reshape exceedance afterwards to bring it back to 3D
        # (n_tiles, sim_size, n_arms)
        y_flat = y.reshape((-1, 3))
        n_flat = jnp.full_like(y_flat, self.n_arm_samples)
        data = jnp.stack((y_flat, n_flat), axis=-1)
        test_stat_per_arm = self.fi.test_inference(data).reshape(y.shape)
        return jnp.min(
            jnp.where(null_truth[:, None, :], test_stat_per_arm, jnp.inf), axis=-1
        )


@dataclass
class QuadRule:
    pts: np.ndarray
    wts: np.ndarray


def gauss_rule(n, a=-1, b=1):
    """
    Points and weights for a Gaussian quadrature with n points on the interval
    (a, b)
    """
    pts, wts = np.polynomial.legendre.leggauss(n)
    pts = (pts + 1) * (b - a) / 2 + a
    wts = wts * (b - a) / 2
    return QuadRule(pts, wts)


def log_gauss_rule(N, a, b):
    A = np.log(a)
    B = np.log(b)
    qr = gauss_rule(N, a=A, b=B)
    pts = np.exp(qr.pts)
    wts = np.exp(qr.pts) * qr.wts
    return QuadRule(pts, wts)


class FastINLA:
    def __init__(
        self,
        n_arms=4,
        mu_0=-1.34,
        mu_sig2=100.0,
        sigma2_n=15,
        sigma2_bounds=(1e-6, 1e3),
        sigma2_alpha=0.0005,
        sigma2_beta=0.000005,
        p1=0.3,
        critical_value=0.85,
        opt_tol=1e-3,
    ):
        import scipy.stats

        self.n_arms = n_arms
        self.mu_0 = mu_0
        self.mu_sig2 = mu_sig2
        self.logit_p1 = jax.scipy.special.logit(p1)

        # For numpy impl:
        self.sigma2_n = sigma2_n
        self.sigma2_rule = log_gauss_rule(self.sigma2_n, *sigma2_bounds)
        self.arms = np.arange(self.n_arms)
        self.cov = np.full((self.sigma2_n, self.n_arms, self.n_arms), self.mu_sig2)
        self.cov[:, self.arms, self.arms] += self.sigma2_rule.pts[:, None]
        self.neg_precQ = -np.linalg.inv(self.cov)
        self.logprecQdet = 0.5 * np.log(np.linalg.det(-self.neg_precQ))
        self.log_prior = scipy.stats.invgamma.logpdf(
            self.sigma2_rule.pts, sigma2_alpha, scale=sigma2_beta
        )
        self.opt_tol = opt_tol
        self.thresh_theta = np.full(
            self.n_arms, jax.scipy.special.logit(0.1) - self.logit_p1
        )
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

    def rejection_inference(self, data):
        _, exceedance, _, _ = self.inference(data)
        return exceedance > self.critical_value

    def test_inference(self, data):
        _, exceedance, _, _ = self.inference(data)
        return 1 - exceedance

    def inference(self, data):
        return self.jax_inference(data)[:4]

    def jax_inference(self, data):
        """
        See the numpy implementation for comments explaining the steps. The
        series of operations is almost identical in the JAX implementation.
        """
        y = jnp.asarray(data[..., 0])
        n = jnp.asarray(data[..., 1])
        theta_max, hess_inv = self.jax_opt_vec(
            y,
            n,
            self.cov_jax,
            self.neg_precQ_jax,
            self.sigma2_pts_jax,
            self.logit_p1,
            self.mu_0,
            self.opt_tol,
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


@jax.jit
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
