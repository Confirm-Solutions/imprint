import numpy as np
import inla

def build_centered_quad_rules(model, data, n=11, w=6, max_sigma2=2.0):
    """
    The ideal range of theta integration will depend heavily on the value of sigma2:
    If sigma2 is very small, then there will not be much variation in esimated
    values of theta. If sigma2 is larger, there will be lots of variation. Thus,
    in order to correctly integrate over the theta dimensions, we need to vary
    the domain of theta integration depending on the value of sigma2. The method
    implemented here is fairly simple: 
    - find the MAP for a given value of sigma2.
    - (very approximately) assume a standard deviation of a normal appx at that MAP.
    - construct a gaussian quadrature rule that would conservatively cover that
      normal approximately by integrating out to 6x the standard deviation
    """
    sigma2_rule = model.sigma2_rule
    x0_info = inla.optimize_x0(model, data[:, None, :], sigma2_rule.pts[None, :, None])
    # Note: in INLA notation, x is used to refer to the Berry "theta"
    thetacenters = x0_info["x"]
    thetastd = np.minimum(
        np.sqrt(sigma2_rule.pts), np.full_like(sigma2_rule.pts, max_sigma2)
    )
    thetamins = thetacenters - w * thetastd[None, :, None]
    thetamaxs = thetacenters + w * thetastd[None, :, None]
    points, weights = np.polynomial.legendre.leggauss(n)
    thetapts = np.transpose(
        thetamins[None, :, :, :]
        + (thetamaxs[None, :, :, :] - thetamins[None, :, :, :])
        * (points[:, None, None, None] * 0.5 + 0.5),
        (1, 2, 3, 0),
    )
    thetawts = (
        weights[None, None, None, :]
        * (thetamaxs[:, :, :, None] - thetamins[:, :, :, None])
        * 0.5
    )
    return thetapts, thetawts


def build_integration_grids(thetapts, sigma2_rule, fixed_dims=dict()):
    """
    Build a giant multidimensional array of integration points for theta and sigma2:

    By default, the centered quadrature rules produced by build_centered_quad_rules will
    be used for each theta dimension. However, in some situations, the end goal
    will be to produce a distribution with a theta variable as one of the
    unintegrated variables. In that case, we need to use a fixed grid for that
    theta regardless of the value of sigma2. To specify that fixed grid, we use,
    for example:
    fixed_dims = {0: theta0_grid, 1: theta1_grid}

    Returns:
    - grids has shape (n_sims, n_sigma2, n_theta0, n_theta1, n_theta2, n_theta3, 5)
    n_theta# will b
    """
    theta = []
    for i in range(thetapts.shape[0]):
        theta.append([])
        for j in range(thetapts.shape[1]):
            meshgrid_entries = []
            for k in range(4):
                if k in fixed_dims:
                    meshgrid_entries.append(fixed_dims[k].pts)
                else:
                    meshgrid_entries.append(thetapts[i, j, k, :])
            theta[i].append(np.meshgrid(*meshgrid_entries, indexing="ij"))
    theta = np.transpose(np.array(theta), (0, 1, 3, 4, 5, 6, 2))
    grids = np.concatenate(
        (
            theta,
            np.broadcast_to(
                np.expand_dims(sigma2_rule.pts, (0, 2, 3, 4, 5)), theta.shape[:-1]
            )[..., None],
        ),
        axis=-1,
    )
    giant_grid_theta = grids[..., :4].reshape((grids.shape[0], -1, 4)).copy()
    giant_grid_sigma2 = grids[..., 4:].reshape((grids.shape[0], -1, 1)).copy()
    return grids, giant_grid_theta, giant_grid_sigma2


def quad_sum(
    joint,
    thetawts=None,
    sigma2_rule=None,
    integrate_sigma2=False,
    integrate_thetas=(),
    fixed_dims=dict(),
):
    """
    By specifying integrate_sigma2=True and integrate_thetas=(1,2,3), this
    function will compute, for example: p(\theta_0 | y)
    The returned integral will not be normalized.
    """
    joint_weighted = joint.copy()
    sum_dims = []
    if integrate_sigma2:
        wts = sigma2_rule.wts
        joint_weighted *= np.expand_dims(wts, (0, 2, 3, 4, 5))
        sum_dims.append(1)
    for i in integrate_thetas:
        if i in fixed_dims:
            wts = fixed_dims[i].wts
            add_dims = [0, 1, 2, 3, 4, 5]
        else:
            wts = thetawts[:, :, i]
            add_dims = [2, 3, 4, 5]
        add_dims.remove(i + 2)
        sum_dims.append(i + 2)
        joint_weighted *= np.expand_dims(wts, add_dims)
    return joint_weighted.sum(axis=tuple(sum_dims))


def integrate(
    model, data, *, integrate_sigma2=False, integrate_thetas=(), fixed_dims=dict()
):
    thetapts, thetawts = build_centered_quad_rules(model, data)
    grids, giant_grid_theta, giant_grid_sigma2 = build_integration_grids(
        thetapts, model.sigma2_rule, fixed_dims=fixed_dims
    )

    joint = np.exp(
        model.log_joint(giant_grid_theta, data[:, None, :], giant_grid_sigma2)
    ).reshape(grids.shape[:-1])
    return quad_sum(
        joint,
        thetawts=thetawts,
        sigma2_rule=model.sigma2_rule,
        integrate_sigma2=integrate_sigma2,
        integrate_thetas=integrate_thetas,
    )

# def exact_posterior_x(model, data):
#     t0_rule = util.simpson_rule(21, -6, 6)
#     p_t0_g_y = exact.integrate(
#         b, data[:1],
#         integrate_sigma2=True,
#         integrate_thetas=(1, 2, 3),
#         fixed_dims={0:t0_rule},
#     )
#     p_t0_g_y /= np.sum(p_t0_g_y * t0_rule.wts)