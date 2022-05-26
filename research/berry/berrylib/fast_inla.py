import berrylib.util as util
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.stats
import torch
from jax.config import config
from scipy.special import logit

# This line is critical for enabling 64-bit floats.
config.update("jax_enable_x64", True)


def np_fast_invert(S_in, d):
    S = np.tile(S_in, (d.shape[0], 1, 1, 1))
    for k in range(d.shape[-1]):
        outer = np.einsum("...i,...j->...ij", S[..., k, :], S[..., :, k])
        offset = d[..., k] / (1 + d[..., k] * S[..., k, k])
        S = S - (offset[..., None, None] * outer)
    return S


def pytorch_fast_invert(S, d):
    for k in range(d.shape[-1]):
        offset = d[..., k] / (1 + d[..., k] * S[..., k, k])
        S -= (offset[..., None, None] * S[..., k, None, :]) * S[..., :, None, k]
    return S


class FastINLA:
    def __init__(
        self,
        n_arms=4,
        sigma2_n=15,
        sigma2_bounds=(1e-6, 1e3),
        p1=0.3,
        critical_value=0.85,
        torch_dtype=torch.float64,
        torch_device="cpu",
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
        self.logprecQdet = 0.5 * np.log(np.linalg.det(-self.neg_precQ))
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

        # For Torch impl:
        self.torch_dtype = torch_dtype
        self.torch_device = torch_device
        self.sigma2_pts_torch = torch.tensor(
            self.sigma2_rule.pts, dtype=self.torch_dtype
        ).to(self.torch_device)
        self.sigma2_wts_torch = torch.tensor(
            self.sigma2_rule.wts, dtype=self.torch_dtype
        ).to(self.torch_device)
        self.cov_torch = torch.tensor(self.cov, dtype=self.torch_dtype).to(
            self.torch_device
        )
        self.neg_precQ_torch = torch.tensor(self.neg_precQ, dtype=self.torch_dtype).to(
            self.torch_device
        )
        self.logprecQdet_torch = torch.tensor(
            self.logprecQdet, dtype=self.torch_dtype
        ).to(self.torch_device)
        self.log_prior_torch = torch.tensor(self.log_prior, dtype=self.torch_dtype).to(
            self.torch_device
        )

    def rejection_inference(self, y, n, method="jax"):
        _, exceedance, _, _ = self.inference(y, n, method)
        return exceedance > self.critical_value

    def inference(self, y, n, method="jax"):
        fncs = dict(
            numpy=self.numpy_inference,
            jax=self.jax_inference,
            cpp=self.cpp_inference,
            pytorch=self.pytorch_inference,
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

    def pytorch_inference(self, y, n, thresh_theta=None):
        if thresh_theta is None:
            thresh_theta = self.thresh_theta
        thresh_theta = torch.tensor(thresh_theta, dtype=self.torch_dtype).to(
            self.torch_device
        )

        torch_y = torch.tensor(y, dtype=self.torch_dtype).to(self.torch_device)
        torch_n = torch.tensor(n, dtype=self.torch_dtype).to(self.torch_device)

        theta_max, hess = pytorch_optimize_mode(
            torch_y,
            torch_n,
            self.cov_torch,
            self.neg_precQ_torch,
            self.mu_0,
            self.logit_p1,
            self.tol,
        )
        logjoint = self.pytorch_log_joint(torch_y, torch_n, theta_max)
        log_sigma2_post = logjoint - 0.5 * torch.log(torch.linalg.det(-hess))
        # log_sigma2_post = logjoint + 0.5 * torch.log(torch.linalg.det(-hess_inv))
        sigma2_post = torch.exp(log_sigma2_post)
        sigma2_post /= torch.sum(sigma2_post * self.sigma2_wts_torch, axis=1)[:, None]

        theta_sigma = torch.empty(
            (y.shape[0], self.sigma2_n, self.n_arms),
            dtype=self.torch_dtype,
            device=self.torch_device,
        )
        for i in range(self.n_arms):
            rhs = [0] * self.n_arms
            rhs[i] = 1.0
            rhs_torch = torch.tensor(
                rhs, dtype=self.torch_dtype, device=self.torch_device
            )
            # PERFORMANCE NOTE: Using torch.linalg.solve is much more stable
            # than inverting the hessian directly in 32-bit. It's likely that
            # there would be a faster way of doing this.
            hess_inv_row = torch.linalg.solve(hess, rhs_torch)
            theta_sigma[:, :, i] = torch.sqrt(-hess_inv_row[:, :, i])
        # Version for float64...
        # theta_sigma = torch.sqrt(torch.diagonal(-hess_inv, dim1=2, dim2=3))
        theta_mu = theta_max

        exceedances = []
        dist = torch.distributions.normal.Normal(0, 1)
        for i in range(self.n_arms):
            exc_sigma2 = 1.0 - dist.cdf(
                (thresh_theta[..., None, i] - theta_mu[..., i]) / theta_sigma[..., i]
            )
            exc = torch.sum(
                exc_sigma2 * sigma2_post * self.sigma2_wts_torch[None, :], axis=1
            )
            exceedances.append(exc)
        return (
            sigma2_post.cpu().numpy(),
            torch.stack(exceedances, axis=-1).cpu().numpy(),
            theta_max.cpu().numpy(),
            theta_sigma.cpu().numpy(),
        )

    def pytorch_log_joint(self, y, n, theta):
        """
        theta is expected to have shape (N, n_sigma2, n_arms)
        """
        theta_m0 = theta - self.mu_0
        theta_adj = theta + self.logit_p1
        exp_theta_adj = torch.exp(theta_adj)
        MM = theta_m0[:, :, :, None].reshape((-1, self.n_arms, 1))
        NN = torch.tile(self.neg_precQ_torch[None, ...], (y.shape[0], 1, 1, 1)).reshape(
            (-1, self.n_arms, self.n_arms)
        )
        quad_term = torch.sum(torch.bmm(NN, MM) * MM, axis=(-2, -1)).reshape(
            (y.shape[0], -1)
        )
        return (
            0.5 * quad_term
            + self.logprecQdet_torch
            + torch.sum(
                theta_adj * y[:, None] - n[:, None] * torch.log(exp_theta_adj + 1),
                axis=-1,
            )
            + self.log_prior_torch
        )


def pytorch_optimize_mode(y, n, cov, neg_precQ, mu_0, logit_p1, tol):
    sigma2_n = neg_precQ.shape[0]
    N, n_arms = y.shape
    theta_max = torch.zeros((N, sigma2_n, n_arms), dtype=y.dtype, device=y.device)
    na = torch.arange(4)

    converged = False
    for i in range(100):
        theta_m0 = theta_max - mu_0
        exp_theta_adj = torch.exp(theta_max + logit_p1)
        C = 1.0 / (exp_theta_adj + 1)
        nCeta = n[:, None] * C * exp_theta_adj

        grad = (
            torch.matmul(neg_precQ[None], theta_m0[:, :, :, None])[..., 0]
            + y[:, None]
            - nCeta
        )

        # diag = nCeta * C
        # Version that only works in float64.
        # hess_inv = pytorch_fast_invert(
        #     -torch.tile(cov[None, ...], (diag.shape[0], 1, 1, 1)),
        #     -diag,
        # )
        # step = -torch.matmul(hess_inv, grad[..., None])[..., 0]

        hess = torch.tile(neg_precQ[None, ...], (N, 1, 1, 1))
        hess[..., na, na] -= nCeta * C

        # Take the full Newton step. The negative sign comes here because we
        # are finding a maximum, not a minimum.

        # PERFORMANCE NOTE: Using pytorch_fast_invert is faster but has some
        # instability with float32 and thus is unsuitable to a GPU
        # implementation.
        step = -torch.linalg.solve(hess, grad)

        theta_max += step

        # We use a step size convergence criterion. This seems empirically
        # sufficient. But, it would be possible to also check gradient norms
        # or other common convergence criteria.
        if torch.max(torch.sum(step**2, dim=-1)) < tol**2:
            converged = True
            break

    if not converged:
        raise RuntimeError("Failed to identify the mode of the joint density.")

    return theta_max, hess


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
