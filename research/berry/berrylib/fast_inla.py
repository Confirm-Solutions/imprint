from dataclasses import dataclass
from typing import Callable

import berrylib.util as util
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.stats
from berrylib.fast_math import jax_fast_invert
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


@dataclass
class FastINLAModel:
    log_joint: Callable
    grad_hess: Callable


def logdet(m):
    """Compute the log of the determinant in a numerically stable way."""
    if isinstance(m, np.ndarray):
        numpy = np
    else:
        numpy = jnp
    sign, logdet = numpy.linalg.slogdet(m)
    return sign * logdet


class FastINLA:
    def __init__(
        self,
        model: FastINLAModel = None,
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
        self.n_arms = n_arms
        self.mu_0 = mu_0
        self.mu_sig2 = mu_sig2
        self.logit_p1 = logit(p1)

        # For numpy impl:
        self.sigma2_n = sigma2_n
        self.sigma2_rule = util.log_gauss_rule(self.sigma2_n, *sigma2_bounds)
        self.arms = np.arange(self.n_arms)
        self.cov = np.full((self.sigma2_n, self.n_arms, self.n_arms), self.mu_sig2)
        self.cov[:, self.arms, self.arms] += self.sigma2_rule.pts[:, None]
        self.neg_precQ = -np.linalg.inv(self.cov)
        self.logprecQdet = 0.5 * np.log(np.linalg.det(-self.neg_precQ))
        self.log_prior = scipy.stats.invgamma.logpdf(
            self.sigma2_rule.pts, sigma2_alpha, scale=sigma2_beta
        )
        self.opt_tol = opt_tol
        self.thresh_theta = np.full(self.n_arms, logit(0.1) - self.logit_p1)
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

        self.model: FastINLAModel = model
        if model is None:

            def log_joint(self, data, theta):
                """
                theta is expected to have shape (N, n_sigma2, n_arms)
                """
                y = data[..., 0]
                n = data[..., 1]
                theta_m0 = theta - self.mu_0
                theta_adj = theta + self.logit_p1
                exp_theta_adj = np.exp(theta_adj)
                return (
                    # NB: this has fairly low accuracy in float32
                    0.5
                    * np.einsum("...i,...ij,...j", theta_m0, self.neg_precQ, theta_m0)
                    + self.logprecQdet
                    + np.sum(
                        theta_adj * y[:, None] - n[:, None] * np.log(exp_theta_adj + 1),
                        axis=-1,
                    )
                    + self.log_prior
                )

            def grad_hess(self, data, theta, arms_opt):
                # These formulas are
                # straightforward derivatives from the Berry log joint density
                # see the log_joint method below
                y = data[..., 0]
                n = data[..., 1]
                na = np.arange(len(arms_opt))
                theta_m0 = theta - self.mu_0
                exp_theta_adj = np.exp(theta + self.logit_p1)
                C = 1.0 / (exp_theta_adj + 1)
                grad = (
                    np.matmul(self.neg_precQ[None], theta_m0[:, :, :, None])[..., 0]
                    + y[:, None]
                    - (n[:, None] * exp_theta_adj) * C
                )[..., arms_opt]

                hess = np.tile(
                    self.neg_precQ[None, ..., arms_opt, :][..., :, arms_opt],
                    (y.shape[0], 1, 1, 1),
                )
                hess[..., na, na] -= (n[:, None] * exp_theta_adj * (C**2))[
                    ..., arms_opt
                ]
                return grad, hess

            self.model = FastINLAModel(log_joint, grad_hess)

    def rejection_inference(self, data, method="jax"):
        _, exceedance, _, _ = self.inference(data, method)
        return exceedance > self.critical_value

    def inference(self, data, method="jax"):
        fncs = dict(
            numpy=self.numpy_inference, jax=self.jax_inference, cpp=self.cpp_inference
        )
        return fncs[method](data)[:4]

    def numpy_inference(self, data, thresh_theta=None):
        """
        Bayesian inference of a basket trial given data with n_arms.

        Returns:
            sigma2_post: The posterior density for each value of the sigma2
                quadrature rule.
            exceedances: The probability of exceeding the threshold for each arm.
            theta_max: the mode of p(theta_i, y, sigma^2)
            theta_sigma: the std dev of a gaussian distribution centered at the
                mode of p(theta_i, y, sigma^2)
            hess_inv: the inverse hessian at the mode of p(theta_i, y, sigma^2)
        """
        if thresh_theta is None:
            thresh_theta = self.thresh_theta

        # TODO: warm start with DB theta ?
        # Step 1) Compute the mode of p(theta, y, sigma^2) holding y and sigma^2 fixed.
        # This is a simple Newton's method implementation.
        theta_max, hess_inv = self.optimize_mode(data)

        # Step 2) Calculate the joint distribution p(theta, y, sigma^2)
        logjoint = self.model.log_joint(self, data, theta_max)

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
                thresh_theta[..., None, i],
                theta_mu[..., i],
                theta_sigma[..., i],
            )
            exc = np.sum(
                exc_sigma2 * sigma2_post * self.sigma2_rule.wts[None, :], axis=1
            )
            exceedances.append(exc)
        return (
            sigma2_post,
            np.stack(exceedances, axis=-1),
            theta_max,
            theta_sigma,
            hess_inv,
        )

    def optimize_mode(self, data, fixed_arm_dim=None, fixed_arm_values=None):
        """
        Find the mode with respect to theta of the model log joint density.

        fixed_arm_dim: we permit one of the theta arms to not be optimized: to
            be "fixed".
        fixed_arm_values: the values of the fixed arm.
        """

        # NOTE: If
        # 1) fixed_arm_values is chosen without regard to the other theta values
        # 2) sigma2 is very small
        # then, the optimization problem will be poorly conditioned and ugly because the
        # chances of t_{arm_idx} being very different from the other theta values is
        # super small with small sigma2
        # I am unsure how severe this problem is. So far, it does not appear to
        # have caused problems, but I left this comment here as a guide in case
        # the problem arises in the future.

        N = data.shape[0]
        arms_opt = list(range(self.n_arms))
        theta_max = np.zeros((N, self.sigma2_n, self.n_arms))

        if fixed_arm_dim is not None:
            arms_opt.remove(fixed_arm_dim)
            theta_max[..., fixed_arm_dim] = fixed_arm_values

        converged = False
        # The joint density is composed of:
        # 1) a quadratic term coming from the theta likelihood
        # 2) a binomial term coming from the data likelihood.
        # We ignore the terms that don't depend on theta since we are
        # optimizing here and constant offsets are irrelevant.
        for i in range(100):

            # Calculate the gradient and hessian.
            grad, hess = self.model.grad_hess(self, data, theta_max, arms_opt)
            hess_inv = np.linalg.inv(hess)

            # Take the full Newton step. The negative sign comes here because we
            # are finding a maximum, not a minimum.
            step = -np.matmul(hess_inv, grad[..., None])[..., 0]
            theta_max[..., arms_opt] += step

            # We use a step size convergence criterion. This seems empirically
            # sufficient. But, it would be possible to also check gradient norms
            # or other common convergence criteria.
            if np.max(np.linalg.norm(step, axis=-1)) < self.opt_tol:
                converged = True
                break

        if not converged:
            raise RuntimeError("Failed to identify the mode of the joint density.")

        return theta_max, hess_inv

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

    def cpp_inference(self, data):
        """
        See the numpy implementation for comments explaining the steps. The
        series of operations is almost identical in the C++ implementation.
        """
        import cppimport

        ext = cppimport.imp("berrylib.fast_inla_ext")
        sigma2_post = np.empty((data.shape[0], self.sigma2_n))
        exceedances = np.empty((data.shape[0], self.n_arms))
        theta_max = np.empty((data.shape[0], self.sigma2_n, self.n_arms))
        theta_sigma = np.empty((data.shape[0], self.sigma2_n, self.n_arms))
        ext.inla_inference(
            sigma2_post,
            exceedances,
            theta_max,
            theta_sigma,
            data[..., 0],
            data[..., 1],
            self.sigma2_rule.pts,
            self.sigma2_rule.wts,
            self.log_prior,
            self.neg_precQ,
            self.cov,
            self.logprecQdet,
            self.mu_0,
            self.logit_p1,
            self.opt_tol,
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


def FastINLASurvival(*, lambdaj, **kwargs):
    """Rough draft of a builder for running INLA on survival data.

    Parameters
    ----------
    lambdaj, optional


    Returns
    -------
        a FastINLA object tailored to survival analysis.
    """
    mu_0 = kwargs["mu_0"]

    @jax.jit
    def log_joint(data, rho, neg_precQ, logprecQdet, log_prior):
        """
        theta is expected to have shape (N, n_sigma2, n_arms)
        """
        n_events = data[..., 0]
        total_obs_time = data[..., 1]
        rho_m0 = rho - mu_0
        hazard = 1.0 / (jnp.exp(rho) * lambdaj[None])
        return (
            0.5 * jnp.einsum("...i,...ij,...j", rho_m0, neg_precQ, rho_m0)
            + logprecQdet
            + jnp.sum(
                jnp.log(hazard) * n_events[:, None] - hazard * total_obs_time[:, None],
                axis=-1,
            )
            + log_prior
        )

    def scalar_log_joint_opt(rho, neg_precQ, n_events, total_obs_time):
        """
        theta is expected to have shape (N, n_sigma2, n_arms)
        """
        rho_m0 = rho - mu_0
        hazard = 1.0 / (jnp.exp(rho) * lambdaj)
        return 0.5 * neg_precQ.dot(rho_m0).T.dot(rho_m0) + jnp.sum(
            jnp.log(hazard) * n_events - hazard * total_obs_time
        )

    grad_opt = jax.jit(
        jax.vmap(
            jax.vmap(jax.grad(scalar_log_joint_opt), in_axes=(0, 0, None, None)),
            in_axes=(0, None, 0, 0),
        )
    )
    hessian_opt = jax.jit(
        jax.vmap(
            jax.vmap(jax.hessian(scalar_log_joint_opt), in_axes=(0, 0, None, None)),
            in_axes=(0, None, 0, 0),
        )
    )

    def grad_hess(fi, data, rho, arms_opt):
        grad = grad_opt(rho, fi.neg_precQ, data[..., 0], data[..., 1])
        hess = hessian_opt(rho, fi.neg_precQ, data[..., 0], data[..., 1])
        return grad, hess

    def log_joint_wrapper(fi, data, rho):
        return log_joint(data, rho, fi.neg_precQ, fi.logprecQdet, fi.log_prior)

    model = FastINLAModel(log_joint_wrapper, grad_hess)
    return FastINLA(model=model, **kwargs)
