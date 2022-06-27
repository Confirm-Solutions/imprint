r"""
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

# Sources of error

The goal of developing the "exact" integrator was to develop more confidence in
our INLA and MCMC estimators. But, first we need to examine the error from this
method.

There are three sources of error:
* theta domain error: we arbitrarily truncate the domain in order to numerically
  integrate. There are other quadrature rules that would handle an infinite
  domain, but they have their own complexities. Fortunately, we can estimate the
  quadrature error resulting from the finite domain by evaluating the integrand
  at the end points of the interval or by expanding the domain. The `tol`
  parameter to build_grid determines the cut off density ratio and smaller tol
  values will construct a larger theta domain.
* approximation error: the default here is to use an 11 point quadrature rule in
  the theta variables. This seems to be sufficient for good accuracy. How fast
  does the Gaussian quadrature here converge? Would 20 points per dimension be a
  lot better? For a smooth integrand, Gaussian quadrature will normally converge
  very quickly.
* $\sigma^2$ domain error: we truncated the range of sharing hyperparameters. This
  affects our density function because we need to normalize by the integral over
  the entire domain of hyperparameters. This $\sigma^2$ domain error can be made
  consistent between the different methods we examine by using the same $\theta$
  grid in the INLA calculations. This error is also the easiest form of error to
  control by a user because it is very obvious when a PDF is still large at the
  edge of the domain. So, we ignore this source of error.

The procedure here is as follows:
1. Construct an "eta-space" grid that is simply a cartesian product of the
    gaussian quadrature rule.
2. Identify the mode and variance of the distribution using the same tools
    created for the INLA implementation.
3. Decompose the variance into an orthogonal coordinate system using an
    eigendecomposition
4. Line search along the axes of the orthogonal coordinate system to
    determine how much of the space to cover with our quadrature grid.
5. Assume the rectangular eta grid is in this orthogonal coordinate system
6. Map from eta to theta via the eigenvectors/values
7. Don't forget to multiply the quadrature weights by the determinant of the
    jacobian of the transformation.
"""
import berrylib.util as util
import numpy as np


def build_grid(
    fi,
    y_in,
    n_in,
    *,
    integrate_sigma2=True,
    fixed_arm_dim=None,
    fixed_arm_values=None,
    n_theta=11,
    w_theta=5,
    tol=1e-4
):
    """
    Build a theta integration grid for each value of sigma2. See the docstring
    for this module for further details on the procedure.

    fi:
        a FastINLA object, used for determining the mode and variance of the
        distribution being integrated.

    y, n:
        Binomial count data!

    integrate_sigma:
        should we integrate over the sigma2 values

    fixed_dims:
        any theta dimensions for which we should use a pre-specified
        list of points instead of determining the optimal range. this is
        specified as a dictionary with integer keys specifying the fixed
        dimensions and a numpy array specifying the points to use.
        NOTE: we will not be integrating over these dimensions.

    n_theta:
        the number of quadrature points per dimension.

    w_theta:
        the number of standard deviations to extend outwards from the mode.

    """
    assert y_in.ndim == 1
    assert n_in.ndim == 1
    R = y_in.shape[0]

    S = fi.sigma2_n
    integrate_thetas = list(range(fi.n_arms))
    if fixed_arm_dim is not None:
        integrate_thetas.remove(fixed_arm_dim)

    etapts, etawts = np.polynomial.legendre.leggauss(n_theta)
    grid_eta = np.stack(
        np.meshgrid(*[etapts for k in integrate_thetas], indexing="ij"), axis=-1
    )
    grid_eta_wts = np.prod(
        np.stack(
            np.meshgrid(*[etawts for k in integrate_thetas], indexing="ij"), axis=-1
        ),
        axis=-1,
    )

    # Compute mode and the inverse hessian at the mode via INLA!
    if fixed_arm_dim is None:
        y = y_in[None, :]
        n = n_in[None, :]
        data = np.stack((y, n), axis=-1)
        mode, hess_inv = fi.optimize_mode(data)
        assert mode.shape[0] == 1
    else:
        M = fixed_arm_values.shape[0]
        y = np.tile(y_in[None, :], (M, 1)).reshape((-1, R))
        n = np.tile(n_in[None, :], (M, 1)).reshape((-1, R))
        data = np.stack((y, n), axis=-1)
        arm_values_tiled = np.tile(fixed_arm_values[:, None], (1, S))
        mode, hess_inv = fi.optimize_mode(
            data,
            fixed_arm_dim=fixed_arm_dim,
            fixed_arm_values=arm_values_tiled,
        )
        np.testing.assert_allclose(mode[..., fixed_arm_dim], arm_values_tiled)
        assert mode.shape[0] == M

    # Step 2: decorrelate our coordinate system.
    # the negative of hess_inv is the covariance matrix. We take the subset
    # corresponding to the dimensions we will be integrating over.
    eigenvals, eigenvecs = np.linalg.eigh(-hess_inv)
    axis_half_len = np.sqrt(np.abs(eigenvals))

    # Step 3: explore the coordinate system axis to determine the necessary
    # domain scale.
    # TODO: this is the slow part of the calculation because:
    #   1) all the for loops
    log_tol = np.log(tol)
    mode_logjoint = fi.model.log_joint(fi, data, mode)

    for eigen_idx in range(len(integrate_thetas)):
        scaled_eigenvecs = (
            axis_half_len[:, :, None, eigen_idx] * eigenvecs[:, :, :, eigen_idx]
        )
        steps = np.empty((y.shape[0], fi.sigma2_n, 2))
        for i, direction in enumerate([-1, 1]):

            def f(x):
                probe = mode.copy()
                probe[..., integrate_thetas] += (
                    scaled_eigenvecs * x[..., None] * direction
                )
                return fi.model.log_joint(fi, data, probe) - mode_logjoint - log_tol

            left = np.zeros((y.shape[0], fi.sigma2_n))
            right = np.full((y.shape[0], fi.sigma2_n), 100.0)
            soln, _, _ = vectorized_bisection(f, left, right)
            steps[:, :, i] = direction * soln
        mode_shift = np.mean(steps, axis=-1)
        mode[:, :, integrate_thetas] += scaled_eigenvecs * mode_shift[..., None]
        widths = steps[..., 1] - mode_shift
        assert np.all(widths > 0)
        axis_half_len[:, :, eigen_idx] *= widths

    # Map the coordinates from eta to theta space.
    mode_subset = mode[..., integrate_thetas]
    broadcast_shape = list(mode_subset.shape)
    for i in range(len(integrate_thetas)):
        broadcast_shape.insert(1, 1)
    grid_theta = np.einsum(
        "klji,...i,kli->k...lj", eigenvecs, grid_eta, axis_half_len
    ) + mode_subset.reshape(broadcast_shape)

    # Map the quadrature weights from eta to theta space.
    # We need to multiply by the absolute value of the determinant of the
    # transformation. Because we have already computed a eigendecomposition, the
    # determinant of matrix is just product of eigenvalues.
    det_jacobian = axis_half_len.prod(axis=-1)
    grid_theta_wts = np.einsum("kl,...->k...l", det_jacobian, grid_eta_wts)

    # Reshape the final results to:
    # (N, n_theta1, ..., n_thetaM, n_sigma2, n_arms + 1)
    # The entries corresponding to theta_i will be full_grid[..., i]
    # while the entries corresponding to sigma2 will be full_grid[..., -1]
    full_grid = np.empty((*grid_theta.shape[:-1], R + 1))
    full_grid[..., integrate_thetas] = grid_theta
    full_grid[..., fi.n_arms] = util.broadcast(
        fi.sigma2_rule.pts, full_grid.shape[:-1], [-1]
    )
    full_wts = grid_theta_wts
    if fixed_arm_dim is not None:
        final_shape = [M] + list(full_grid.shape[1:])
        full_grid = full_grid.reshape(final_shape)
        full_grid[..., fixed_arm_dim] = util.broadcast(
            fixed_arm_values, full_grid.shape[:-1], [0]
        )
        full_wts = grid_theta_wts.reshape(final_shape[:-1])
    else:
        full_grid = full_grid[0]
        full_wts = full_wts[0]

    if integrate_sigma2:
        full_wts *= util.broadcast(fi.sigma2_rule.wts, full_wts.shape, [-1])

    # Calculate joint density at every grid point.
    grids_ravel = full_grid[..., : fi.n_arms].reshape((-1, fi.sigma2_n, fi.n_arms))

    n_theta_pts = np.prod(full_grid.shape[0 : fi.n_arms])
    y_tiled = np.tile(y_in[None], (n_theta_pts, 1)).reshape((-1, fi.n_arms))
    n_tiled = np.tile(n_in[None], (n_theta_pts, 1)).reshape((-1, fi.n_arms))
    logjoint = fi.model.log_joint(
        fi, np.stack((y_tiled, n_tiled), axis=-1), grids_ravel
    ).reshape(full_grid.shape[:-1])

    return full_grid, full_wts, logjoint


def vectorized_bisection(f, left, right, max_iter=100, tol=1e-4):
    A = f(left)
    B = f(right)
    assert np.all(A > 0)
    assert np.all(B < 0)
    for j in range(max_iter):
        new = (left + right) * 0.5
        F = f(new)
        if np.all(np.abs(F) < tol):
            break
        greater_idxs = np.where(F > 0)
        left[greater_idxs] = new[greater_idxs]
        A[greater_idxs] = F[greater_idxs]

        lesser_idxs = np.where(F <= 0)
        right[lesser_idxs] = new[lesser_idxs]
        B[lesser_idxs] = F[lesser_idxs]
    return new, F, j


def integrate(
    fi,
    y,
    n,
    *,
    integrate_sigma2=True,
    fixed_arm_dim=None,
    fixed_arm_values=None,
    n_theta=11,
    w_theta=6,
    return_intermediates=False,
    tol=1e-4
):
    grids, wts, logjoint = build_grid(
        fi,
        y,
        n,
        integrate_sigma2=integrate_sigma2,
        fixed_arm_dim=fixed_arm_dim,
        fixed_arm_values=fixed_arm_values,
        n_theta=n_theta,
        w_theta=w_theta,
        tol=tol,
    )

    joint = np.exp(logjoint)

    if fixed_arm_dim is None:
        sum_dims = list(range(0, fi.n_arms))
    else:
        sum_dims = list(range(1, fi.n_arms))

    if integrate_sigma2:
        sum_dims.append(-1)

    if return_intermediates:
        return np.sum(joint * wts, tuple(sum_dims)), grids, wts, joint
    else:
        return np.sum(joint * wts, tuple(sum_dims))


def quadrature_posterior_theta(fi, y, n, thresh):
    theta_map = np.empty_like(thresh)
    cilow = np.empty_like(thresh)
    cihi = np.empty_like(thresh)
    exceedance = np.empty_like(thresh)
    t_rule = util.simpson_rule(61, -6.0, 1.0)
    for j in range(thresh.shape[0]):
        for i in range(4):
            integrate_dims = list(range(4))
            integrate_dims.remove(i)
            p_ti_g_y = integrate(
                fi,
                y[j],
                n[j],
                integrate_sigma2=True,
                fixed_arm_dim=i,
                fixed_arm_values=t_rule.pts,
            )
            p_ti_g_y /= np.sum(p_ti_g_y * t_rule.wts)

            cdf = []
            cdf_pts = []
            # TODO: build custom product rule for each step! That would be much more
            # accurate.
            for k in range(3, t_rule.pts.shape[0], 2):
                # Note that t0_rule.wts[:i] will be different from cdf_rule.wts!!
                cdf_rule = util.simpson_rule(k, t_rule.pts[0], t_rule.pts[k - 1])
                cdf.append(np.sum(p_ti_g_y[:k] * cdf_rule.wts[:k]))
                cdf_pts.append(t_rule.pts[k - 1])
            cdf = np.array(cdf).T
            cdf_pts = np.array(cdf_pts)

            # TODO: I should do a linear interpolation here too.
            cilow[j, i] = cdf_pts[np.argmax(cdf > 0.025)]
            cihi[j, i] = cdf_pts[np.argmax(cdf > 0.975)]
            theta_map[j, i] = t_rule.pts[np.argmax(p_ti_g_y)]

            above_idx = np.argmax(cdf_pts > thresh[j, i, None])
            below_idx = above_idx - 1
            a = cdf_pts[below_idx]
            b = cdf_pts[above_idx]
            b_mult = (thresh[j, i] - a) / (b - a)
            a_mult = 1 - b_mult

            interp_cdf = a_mult * cdf[below_idx] + b_mult * cdf[above_idx]
            exceedance[j, i] = 1.0 - interp_cdf

    return dict(cilow=cilow, cihi=cihi, theta_map=theta_map, exceedance=exceedance)
