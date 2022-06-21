from functools import partial

import berrylib.util as util
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.stats
from jax.config import config
from scipy.special import logit

# This line is critical for enabling 64-bit floats.
config.update("jax_enable_x64", True)


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
        n_arms=4,
        sigma2_n=15,
        sigma2_bounds=(1e-6, 1e3),
        p1=0.3,
        critical_value=0.85,
    ):
        self.n_arms = n_arms
        self.mu_0 = -1.34
        self.mu_sig_sq = 100.0
        self.logit_p1 = logit(p1)

        # For numpy impl:
        self.sigma2_n = sigma2_n
        self.sigma2_rule = util.log_gauss_rule(self.sigma2_n, *sigma2_bounds)
        self.arms = np.arange(self.n_arms)
        self.cov = np.full((self.sigma2_n, self.n_arms, self.n_arms), self.mu_sig_sq)
        self.cov[:, self.arms, self.arms] += self.sigma2_rule.pts[:, None]
        self.neg_precQ = -np.linalg.inv(self.cov)
        self.logprecQdet = 0.5 * logdet(-self.neg_precQ)
        self.precQ_eig_vals, self.precQ_eig_vecs = np.linalg.eigh(-self.neg_precQ)
        assert not np.isnan(self.precQ_eig_vecs).any()

        self.log_prior = scipy.stats.invgamma.logpdf(
            self.sigma2_rule.pts, 0.0005, scale=0.000005
        )
        self.tol = 1e-3
        self.thresh_theta = np.full(self.n_arms, logit(0.1) - self.logit_p1)
        self.critical_value = critical_value

        # For JAX impl:
        self.sigma2_pts_jax = jnp.asarray(self.sigma2_rule.pts)
        self.sigma2_wts_jax = jnp.asarray(self.sigma2_rule.wts)
        self.cov_jax = jnp.asarray(self.cov)
        self.neg_precQ_jax = jnp.asarray(self.neg_precQ)
        self.precQ_eig_vals_jax = jnp.asarray(self.precQ_eig_vals)
        self.precQ_eig_vecs_jax = jnp.asarray(self.precQ_eig_vecs)
        self.logprecQdet_jax = jnp.asarray(self.logprecQdet)
        self.log_prior_jax = jnp.asarray(self.log_prior)

        self.jax_opt_vec = jax.jit(
            jax.vmap(
                jax.vmap(
                    jax_opt,
                    in_axes=(None, None, 0, 0, 0, 0, None, None, None),
                    out_axes=(0, 0, 0),
                ),
                in_axes=(0, 0, None, None, None, None, None, None, None),
                out_axes=(0, 0, 0),
            )
        )

    def rejection_inference(self, y, n, method="jax"):
        _, exceedance, _, _ = self.inference(y, n, method)
        return exceedance > self.critical_value

    def inference(self, y, n, method="jax"):
        fncs = dict(
            numpy=self.numpy_inference, jax=self.jax_inference, cpp=self.cpp_inference
        )
        return fncs[method](y, n)[:4]

    def numpy_inference(self, y, n, thresh_theta=None):
        """
        Bayesian inference of the Berry model given data (y, n) with n_arms.

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
        theta_max, hess_inv = self.optimize_mode(y, n)

        # Step 2) Calculate the joint distribution p(theta, y, sigma^2)
        logjoint = self.log_joint(y, n, theta_max)

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

    def optimize_mode(self, y, n, fixed_arm_dim=None, fixed_arm_values=None):
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

        N = y.shape[0]
        arms_opt = list(range(self.n_arms))
        theta_max = np.zeros((N, self.sigma2_n, self.n_arms))
        na = np.arange(self.n_arms)

        if fixed_arm_dim is not None:
            na = np.arange(self.n_arms - 1)
            arms_opt.remove(fixed_arm_dim)
            theta_max[..., fixed_arm_dim] = fixed_arm_values

        converged = False
        # The joint density is composed of:
        # 1) a quadratic term coming from the theta likelihood
        # 2) a binomial term coming from the data likelihood.
        # We ignore the terms that don't depend on theta since we are
        # optimizing here and constant offsets are irrelevant.
        for i in range(100):

            # Calculate the gradient and hessian. These formulas are
            # straightforward derivatives from the Berry log joint density
            # see the log_joint method below
            theta_m0 = theta_max - self.mu_0
            exp_theta_adj = np.exp(theta_max + self.logit_p1)
            C = 1.0 / (exp_theta_adj + 1)
            grad = (
                np.matmul(self.neg_precQ[None], theta_m0[:, :, :, None])[..., 0]
                + y[:, None]
                - (n[:, None] * exp_theta_adj) * C
            )[..., arms_opt]

            hess = np.tile(
                self.neg_precQ[None, ..., arms_opt, :][..., :, arms_opt], (N, 1, 1, 1)
            )
            hess[..., na, na] -= (n[:, None] * exp_theta_adj * (C**2))[..., arms_opt]
            hess_inv = np.linalg.inv(hess)

            # Take the full Newton step. The negative sign comes here because we
            # are finding a maximum, not a minimum.
            step = -np.matmul(hess_inv, grad[..., None])[..., 0]
            theta_max[..., arms_opt] += step

            # We use a step size convergence criterion. This seems empirically
            # sufficient. But, it would be possible to also check gradient norms
            # or other common convergence criteria.
            if np.max(np.linalg.norm(step, axis=-1)) < self.tol:
                converged = True
                break

        if not converged:
            raise RuntimeError("Failed to identify the mode of the joint density.")

        return theta_max, hess_inv

    def log_joint(self, y, n, theta):
        """
        theta is expected to have shape (N, n_sigma2, n_arms)
        """
        theta_m0 = theta - self.mu_0
        theta_adj = theta + self.logit_p1
        exp_theta_adj = np.exp(theta_adj)
        return (
            # NB: this has fairly low accuracy in float32
            0.5 * np.einsum("...i,...ij,...j", theta_m0, self.neg_precQ, theta_m0)
            + self.logprecQdet
            + np.sum(
                theta_adj * y[:, None] - n[:, None] * np.log(exp_theta_adj + 1),
                axis=-1,
            )
            + self.log_prior
        )

    def jax_inference(self, y, n):
        """
        See the numpy implementation for comments explaining the steps. The
        series of operations is almost identical in the JAX implementation.
        """
        y = jnp.asarray(y)
        n = jnp.asarray(n)
        num_iters, theta_max, hess_inv = self.jax_opt_vec(
            y,
            n,
            self.sigma2_pts_jax,
            self.neg_precQ_jax,
            self.precQ_eig_vals_jax,
            self.precQ_eig_vecs_jax,
            self.logit_p1,
            self.mu_0,
            self.tol,
        )

        sigma2_post, exceedances, theta_sigma = jax_calc_posterior_and_exceedances(
            theta_max,
            y,
            n,
            self.log_prior_jax,
            self.precQ_eig_vals_jax,
            self.precQ_eig_vecs_jax,
            hess_inv,
            self.sigma2_wts_jax,
            self.logit_p1,
            self.mu_0,
            self.thresh_theta,
        )

        return sigma2_post, exceedances, theta_max, theta_sigma

    def cpp_inference(self, y, n):
        """
        See the numpy implementation for comments explaining the steps. The
        series of operations is almost identical in the C++ implementation.
        """
        import cppimport

        ext = cppimport.imp("berrylib.fast_inla_ext")
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


def jax_opt(
    y,
    n,
    sigma2,
    neg_precQ,
    prec_eig_vals,
    prec_eig_vecs,
    logit_p1,
    mu_0,
    tol,
    max_iter=100,
    fast_loop=True,
):
    cov = jnp.full_like(neg_precQ, 100) + jnp.diag(jnp.repeat(sigma2, len(y)))

    def step(args):
        i, theta_max, hess_inv, go = args
        theta_m0 = theta_max - mu_0
        exp_theta_adj = jnp.exp(theta_max + logit_p1)
        C = 1.0 / (exp_theta_adj + 1)
        nCeta = n * C * exp_theta_adj

        grad = neg_precQ.dot(theta_m0) + y - nCeta
        diag = nCeta * C

        hess_inv = jax_fast_inv(-cov, -diag)
        step = -hess_inv.dot(grad)
        go = jnp.sum(step**2) > tol**2
        return i + 1, theta_max + step, hess_inv, go

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
    init_args = (jnp.int16(0), theta_max0, jnp.zeros_like(cov), True)

    if fast_loop:
        out = jax.lax.while_loop(
            lambda args: ((args[0] < max_iter) & args[-1]), step, init_args
        )
    else:
        args = init_args
        # step = jax.jit(step)
        for i in range(max_iter):
            args = step(args)
            out = args
            if not args[-1]:
                break
    i, theta_max, hess_inv, go = out
    return i, theta_max, hess_inv


def jax_fast_inv(S, d):
    for k in range(d.shape[0]):
        offset = d[k] / (1 + d[k] * S[k, k])
        S = S - (offset * (S[k, None, :] * S[:, None, k]))
    return S


def jax_faster_inv(D, S):
    """Compute the inverse of a diagonal matrix D plus a shift S.

    This function uses "Sherman-Morrison" formula:
    https://en.wikipedia.org/wiki/Sherman–Morrison_formula
    """
    D_inverse = 1.0 / D
    # NB: reusing D_inverse in this line is numerically unstable
    multiplier = -S / (1 + (S / D).sum())
    M = multiplier * jnp.outer(D_inverse, D_inverse)
    M = M + jnp.diag(D_inverse)
    return M


def jax_faster_inv_diag(D, S):
    """Compute the diagonal of the inverse of a diagonal matrix D plus a shift S.

    This function uses "Sherman-Morrison" formula:
    https://en.wikipedia.org/wiki/Sherman–Morrison_formula
    """
    D_inverse = 1.0 / D
    # NB: reusing D_inverse in this line is numerically unstable
    multiplier = -S / (1 + (S / D).sum())
    return multiplier * D_inverse * D_inverse + D_inverse


def jax_faster_inv_product(D, S, G):
    """Compute (diag(D)+S)^-1 @ G.

    This function uses "Sherman-Morrison" formula:
    https://en.wikipedia.org/wiki/Sherman–Morrison_formula
    """
    D_norm = jnp.abs(D).sum()
    D_normed = D / D_norm
    return (-S * (G / D_normed).sum() / (D_norm + (S / D_normed).sum()) + G) / D


def jax_faster_log_det(D, S):
    """Compute the log determinant of a diagnal matrix D plus a shift S.

    Valid only if the determinant is positive.

    This function uses "Sherman-Morrison for determinants"
    https://en.wikipedia.org/wiki/Matrix_determinant_lemma
    """
    detD_inverse = jnp.log(D).sum()
    newdeterminant = detD_inverse + jnp.log1p((S / D).sum())
    return newdeterminant


def log_normal_pdf(x, mean, prec_eig_vals, prec_eig_vecs, omit_constants=True):
    """Compute the log of the multivariate normal pdf.

    Using the eigendecomposition is numerically stable. Adapted from scipy.
    """
    logdet = -jnp.sum(jnp.log(prec_eig_vals))
    U = prec_eig_vecs * jnp.sqrt(prec_eig_vals)
    dev = x - mean
    # "maha" for "Mahalanobis distance".
    maha = jnp.square(jnp.dot(dev, U)).sum()
    if omit_constants:
        return -0.5 * (maha + logdet)
    else:
        rank = len(prec_eig_vals)
        log2pi = jnp.log(2 * jnp.pi)
        return -0.5 * (rank * log2pi + maha + logdet)


@partial(
    jax.jit,
    static_argnames=[
        "logit_p1",
        "mu_0",
    ],
)
def jax_calc_posterior_and_exceedances(
    theta_max,
    y,
    n,
    log_prior,
    precQ_eig_vals,
    precQ_eig_vecs,
    hess_inv,
    sigma2_wts,
    logit_p1,
    mu_0,
    thresh_theta,
):
    theta_adj = theta_max + logit_p1
    exp_theta_adj = jnp.exp(theta_adj)

    @partial(jnp.vectorize, signature="(k),(l),(l,l)->()")
    def pdf(x, vals, vecs):
        return log_normal_pdf(x, mu_0, vals, vecs)

    logjoint = (
        pdf(theta_max, precQ_eig_vals, precQ_eig_vecs)
        + jnp.sum(
            theta_adj * y[:, None] - n[:, None] * jnp.log(exp_theta_adj + 1),
            axis=-1,
        )
        + log_prior
    )
    logdet_hess_inv = jax.vmap(jax.vmap(logdet))(-hess_inv)
    log_sigma2_post = logjoint + 0.5 * logdet_hess_inv

    # This helps prevent underflow
    log_sigma2_post -= jnp.nanmin(log_sigma2_post, axis=1)[:, None]
    sigma2_post = jnp.exp(log_sigma2_post)
    sigma2_post = jnp.nan_to_num(sigma2_post, posinf=0, neginf=0)
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
