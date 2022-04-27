"""
# Multidimensional numerical integration

Implementation of a numerical integration solver for low dimensional models.
This runs on the Berry model.

We'd like to verify our INLA implementation. MCMC is an alternative that would
give a rough verification. However, MCMC converges quite slowly so it's
difficult to get nearly exact posterior estimates. Fortunately, the example
problem that we are considering here has only five dimensions: one
hyperparameter controlling the sharing and four latent variables that describe
the success of each trial arm. A five dimensional integral is feasible to
compute numerically.

A fully robust implementation would use adaptive quadrature strategies and cover
a much larger parameter domain. However, our goal is not to build a robust multi
dimensional integration tool. Instead, our goal is to simply demonstrate that
the marginal distributions being produced from our INLA implementation are
reasonably good.

# Sources of error

The goal of developing the "exact" integrator was to develop more confidence in
our INLA and MCMC estimators. But, first we need to examine the error from this
method.

There are three sources of error:
* $x$ domain error: we arbitrarily truncated the domain in order to numerically
  integrate. There are other quadrature rules that would handle an infinite
  domain, but they have their own complexities. Fortunately, we can estimate the
  quadrature error resulting from the finite domain by evaluating the integrand
  at the end points of the interval.
* approximation error: we chose to use an 11 point quadrature rule in the latent
  variables. How much more accurate would the integral be if we had chosen a 12
  point quadrature rule? Or a 20 point rule? For a smooth integrand, Gaussian
  quadrature will normally converge very quickly.
* $\theta$ domain error: we truncated the range of sharing hyperparameters. This
  affects our density function because we need to normalize by the integral over
  the entire domain of hyperparameters. This $\theta$ domain error can be made
  consistent between the different methods we examine by using the same $\theta$
  grid in the INLA calculations. This error is also the easiest form of error to
  control by a user because it is very obvious when a PDF is still large at the
  edge of the domain. So, we ignore this source of error.
"""
# import inla
import numpy as np
import util


def build_grid(
    fi, y, n, *, integrate_sigma=True, integrate_thetas=None, n_theta=11, w_theta=6
):
    points, weights = np.polynomial.legendre.leggauss(n_theta)
    etapts = w_theta * points
    etawts = w_theta * weights
    grid_eta = np.stack(
        np.meshgrid(*[etapts for k in range(fi.n_arms)], indexing="ij"), axis=-1
    )
    grid_eta_wts = np.prod(
        np.stack(
            np.meshgrid(*[etawts for k in range(fi.n_arms)], indexing="ij"), axis=-1
        ),
        axis=-1,
    )

    phat = y / n
    sample_I = n * phat * (1 - phat)

    # TODO: we can also use inla hess_inv to compute the full variance here with
    # higher fidelity than DB.
    _, _, mode, _, hess_inv = fi.numpy_inference(y, n)
    sigma_posterior = hess_inv  # np.transpose(hess_inv, (0, 1, 3, 2))

    # Dirty bayes calculation of variance matrix.
    sigma_precision = np.empty((sample_I.shape[0], *fi.neg_precQ.shape))
    sigma_precision[:] = fi.neg_precQ[None, ...]
    sigma_precision[..., fi.arms, fi.arms] += sample_I[:, None, ...]
    sigma_posterior2 = np.linalg.inv(sigma_precision)

    # Step : decorrelate our coordinate system.
    w, v = np.linalg.eigh(sigma_posterior)
    std_dev_eig = w  # np.sqrt(np.abs(w))

    # Map from eta to theta space.
    broadcast_shape = list(mode.shape)
    for i in range(fi.n_arms):
        broadcast_shape.insert(1, 1)
    grid_theta = np.einsum(
        "klij,...i,kli->k...lj", v, grid_eta, std_dev_eig
    ) + mode.reshape(broadcast_shape)

    # We need to multiply by the absolute value of the determinant of the
    # transformation. Because we have already computed a eigendecomposition, the
    # determinant of matrix is just product of eigenvalues.
    det_jacobian = std_dev_eig.prod(axis=-1)
    grid_theta_wts = np.einsum("kl,ij->kijl", det_jacobian, grid_eta_wts)

    sigma2_broadcast = [1] * len(grid_theta_wts.shape)
    sigma2_broadcast[-1] = fi.sigma2_rule.wts.shape[0]
    full_grid = np.empty((*grid_theta_wts.shape, fi.n_arms + 1))
    full_grid[..., :2] = grid_theta
    full_grid[..., 2] = fi.sigma2_rule.pts.reshape(sigma2_broadcast)

    full_wts = grid_theta_wts
    if integrate_sigma:
        full_wts *= fi.sigma2_rule.wts.reshape(sigma2_broadcast)

    return grid_theta, full_wts
    # return full_grid, full_wts


def quad_sum(
    joint,
    thetawts=None,
    sigma2_rule=None,
    integrate_sigma2=False,
    integrate_thetas=(),
    n_arms=4,
    fixed_dims=dict(),
):
    """
    By specifying integrate_sigma2=True and integrate_thetas=(1,2,3), this
    function will compute, for example: p(\theta_0 | y)
    The returned integral will not be normalized.
    """
    joint_weighted = joint.copy()
    sum_dims = []
    theta_dims = range(2, 2 + n_arms)
    if integrate_sigma2:
        wts = sigma2_rule.wts
        joint_weighted *= np.expand_dims(wts, (0, *theta_dims))
        sum_dims.append(1)
    for i in integrate_thetas:
        if i in fixed_dims:
            wts = fixed_dims[i].wts
            add_dims = [0, 1, *theta_dims]
        else:
            wts = thetawts[:, :, i]
            add_dims = list(theta_dims)
        add_dims.remove(i + 2)
        sum_dims.append(i + 2)
        joint_weighted *= np.expand_dims(wts, add_dims)
    return joint_weighted.sum(axis=tuple(sum_dims))


def integrate(
    fi,
    y,
    n,
    *,
    integrate_sigma2=False,
    integrate_thetas=(),
    fixed_dims=dict(),
    n_theta=11,
    w_theta=6,
):
    pts, wts = build_grid(fi, y, n, n_quad=n_theta, w_quad=w_theta)

    joint = np.exp(
        fi.log_joint(giant_grid_theta, data[:, None, :], giant_grid_sigma2)
    ).reshape(grids.shape[:-1])
    return quad_sum(
        joint,
        thetawts=thetawts,
        sigma2_rule=model.sigma2_rule,
        integrate_sigma2=integrate_sigma2,
        integrate_thetas=integrate_thetas,
        n_arms=n_arms,
    )


def quadrature_posterior_theta(model, data, thresh):
    theta_map = np.empty_like(thresh)
    cilow = np.empty_like(thresh)
    cihi = np.empty_like(thresh)
    exceedance = np.empty_like(thresh)
    t_rule = util.simpson_rule(61, -6.0, 1.0)
    for i in range(4):
        integrate_dims = list(range(4))
        integrate_dims.remove(i)
        p_ti_g_y = integrate(
            model,
            data,
            integrate_sigma2=True,
            integrate_thetas=integrate_dims,
            fixed_dims={i: t_rule},
        )
        p_ti_g_y /= np.sum(p_ti_g_y * t_rule.wts, axis=1)[:, None]

        cdf = []
        cdf_pts = []
        # TODO: build custom product rule for each step!
        for j in range(3, t_rule.pts.shape[0], 2):
            # Note that t0_rule.wts[:i] will be different from cdf_rule.wts!!
            cdf_rule = util.simpson_rule(j, t_rule.pts[0], t_rule.pts[j - 1])
            cdf.append(np.sum(p_ti_g_y[:, :j] * cdf_rule.wts[:j], axis=1))
            cdf_pts.append(t_rule.pts[j - 1])
        cdf = np.array(cdf).T
        cdf_pts = np.array(cdf_pts)

        # TODO: I should do a linear interpolation here too
        cilow[:, i] = cdf_pts[np.argmax(cdf > 0.025, axis=1)]
        cihi[:, i] = cdf_pts[np.argmax(cdf > 0.975, axis=1)]
        theta_map[:, i] = t_rule.pts[np.argmax(p_ti_g_y, axis=1)]

        above_idx = np.argmax(cdf_pts > thresh[:, i, None], axis=1)
        below_idx = above_idx - 1
        a = cdf_pts[below_idx]
        b = cdf_pts[above_idx]
        b_mult = (thresh[:, i] - a) / (b - a)
        a_mult = 1 - b_mult

        idxs = np.arange(below_idx.shape[0])
        interp_cdf = a_mult * cdf[idxs, below_idx] + b_mult * cdf[idxs, above_idx]
        exceedance[:, i] = 1.0 - interp_cdf

    return dict(cilow=cilow, cihi=cihi, theta_map=theta_map, exceedance=exceedance)
