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
import numpy as np
import scipy.optimize
import util


def build_grid(
    fi,
    y,
    n,
    *,
    integrate_sigma2=True,
    fixed_arm_dim=None,
    fixed_arm_values=None,
    n_theta=11,
    w_theta=5,
    tol=1e-4
):
    """
    Build a theta integration grid for each value of sigma2.

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

    The procedure here is as follows:
    1. Construct an "eta-space" grid that is simply a cartesian product of the
       gaussian quadrature rule.
    2. Identify the mode and variance of the distribution using the same tools
       created for the INLA implementation.
    3. Decompose the variance into an orthogonal coordinate system using an
       eigendecomposition
    4. Explore the axes of the orthogonal coordinate system to determine how
       much of the space to cover with our quadrature grid.
    5. Assume the rectangular eta grid is in this orthogonal coordinate system
    6. Map from eta to theta via the eigenvectors/values
    7. Don't forget to multiply the quadrature weights by the determinant of the
       jacobian of the transformation.
    """
    N, R = y.shape
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
        y_reshaped, n_reshaped = y, n
        mode, hess_inv = fi.optimize_mode(y, n)
    else:
        M = fixed_arm_values.shape[0]
        y_reshaped = np.tile(y[:, None, :], (1, M, 1)).reshape((-1, R))
        n_reshaped = np.tile(n[:, None, :], (1, M, 1)).reshape((-1, R))
        arm_values_tiled = np.tile(fixed_arm_values[None, :, None], (N, 1, S)).reshape(
            (-1, S)
        )
        mode, hess_inv = fi.optimize_mode(
            y_reshaped,
            n_reshaped,
            fixed_arm_dim=fixed_arm_dim,
            fixed_arm_values=arm_values_tiled,
        )
        np.testing.assert_allclose(mode[..., fixed_arm_dim], arm_values_tiled)

    # Step 2: decorrelate our coordinate system.
    # the negative of hess_inv is the covariance matrix. We take the subset
    # corresponding to the dimensions we will be integrating over.
    w, v = np.linalg.eigh(-hess_inv)
    axis_half_len = np.sqrt(np.abs(w))

    # Step 3: explore the coordinate system axis to determine the necessary
    # domain scale.
    # TODO: this is currently WRONG when more than one problem is passed at a time.
    # the correct solution might be to just change this whole function to work
    # on a single data point
    mode_logjoint = fi.log_joint(y_reshaped, n_reshaped, mode)
    log_tol = np.log(tol)
    for sig_idx in range(fi.sigma2_n):
        for eigen_idx in range(len(integrate_thetas)):
            dir_steps = np.empty((2, *mode.shape[:2]))
            for i, direction in enumerate([-1, 1]):

                def f(x):
                    probe = mode[0, sig_idx].copy()
                    probe[..., integrate_thetas] += (
                        axis_half_len[..., eigen_idx, None]
                        * v[..., :, eigen_idx]
                        * x
                        * direction
                    )
                    return (
                        fi.log_joint(y_reshaped, n_reshaped, probe)[0, sig_idx]
                        - mode_logjoint[0, sig_idx]
                        - log_tol
                    )

                scipy.optimize.bisect(f, 0, 100)
            # for j in range(1, 30)[::-1]:
            #     probe = mode.copy()
            #     probe[..., integrate_thetas] += (
            #         axis_half_len[..., eigen_idx, None]
            #         * v[..., :, eigen_idx]
            #         * j
            #         * direction
            #     )
            #     logjoint = fi.log_joint(y_reshaped, n_reshaped, probe)
            #     delta = logjoint - mode_logjoint
            #     good_steps = np.where(delta < log_tol)
            #     dir_steps[i, good_steps[0], good_steps[1]] = j * direction
        mult = np.max(np.abs(dir_steps), axis=0)
        axis_half_len[..., eigen_idx] *= np.maximum(mult, w_theta)
        # axis_half_len[..., eigen_idx] *= w_theta

    # Map the coordinates from eta to theta space.
    for i in range(10):
        retry = False
        mode_subset = mode[..., integrate_thetas]
        broadcast_shape = list(mode_subset.shape)
        for i in range(len(integrate_thetas)):
            broadcast_shape.insert(1, 1)
        grid_theta = np.einsum(
            "klij,...i,kli->k...lj", v, grid_eta, axis_half_len
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
            final_shape = [N, M] + list(full_grid.shape[1:])
            full_grid = full_grid.reshape(final_shape)
            full_grid[..., fixed_arm_dim] = util.broadcast(
                fixed_arm_values, full_grid.shape[:-1], [1]
            )
            full_wts = grid_theta_wts.reshape(final_shape[:-1])

        if integrate_sigma2:
            full_wts *= util.broadcast(fi.sigma2_rule.wts, full_wts.shape, [-1])

        # Do a final check along the surfaces of our hypercube to make sure that
        # the integrand values are small enough
        # TODO: faster to just calculate the surface values until the domain is correct.
        grids_ravel = full_grid[..., : fi.n_arms].reshape((-1, fi.sigma2_n, fi.n_arms))
        n_theta_pts = np.prod(full_grid.shape[1 : (1 + fi.n_arms)])
        y_tiled = np.tile(y[:, None], (1, n_theta_pts)).reshape((-1, fi.n_arms))
        n_tiled = np.tile(n[:, None], (1, n_theta_pts)).reshape((-1, fi.n_arms))
        logjoint = fi.log_joint(y_tiled, n_tiled, grids_ravel).reshape(
            full_grid.shape[:-1]
        )

        # theta_dims = list(range(1 + fi.n_arms - len(integrate_thetas), 1 + fi.n_arms))
        # theta_dims = tuple(theta_dims)
        # maxv = np.expand_dims(logjoint.max(axis=theta_dims), theta_dims[:-1])
        # for dim in range(len(integrate_thetas)):
        #     fail = np.ones(axis_half_len.shape[:2], dtype=np.bool_)
        #     for dir in [-1, 1]:
        #         idx = [np.s_[:]] * (2 + fi.n_arms)
        #         idx[theta_dims[dim]] = dir
        #         idx = tuple(idx)
        #         fail = fail & np.any(logjoint[idx] - maxv > log_tol, axis=theta_dims[:-1])
        #     if np.any(fail):
        #         # TODO: because axis_half_len is missing the n_data dimension, index fail[0]
        #         # see above for discussion about changing this.
        #         if fixed_arm_dim is not None:
        #             fail = fail[0]
        #         axis_half_len[fail, dim] *= 1.5
        #         retry = True

        if not retry:
            break
    return full_grid, full_wts, logjoint


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
        sum_dims = list(range(1, fi.n_arms + 1))
    else:
        sum_dims = list(range(2, fi.n_arms + 1))

    if integrate_sigma2:
        sum_dims.append(-1)

    if return_intermediates:
        return np.sum(joint * wts, tuple(sum_dims)), grids, wts, joint
    else:
        return np.sum(joint * wts, tuple(sum_dims))


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
