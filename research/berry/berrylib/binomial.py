import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np
import scipy.special
import scipy.stats


def binomial_accumulator(rejection_fnc):
    """
    A simple re-implementation of accumulation. This is useful for distilling
    what is happening during accumulation down to a simple linear sequence of
    operations. Retaining this could be useful for tutorials or conceptual
    introductions to Imprint since we can explain this code without introducing
    most of the framework.

    NOTE: to implement the early stopping procedure from Berry, we will need to
    change all the steps. This function is only valid for a trial with a single
    final analysis.

    theta_tiles: (n_tiles, n_arms), the logit-space parameters for each tile.
    is_null_per_arm: (n_tiles, n_arms), whether each arm's parameter is within
      the null space.
    uniform_samples: (sim_size, n_arm_samples, n_arms), uniform [0, 1] samples
      used to evaluate binomial count samples.
    """

    # We wrap and return this function since rejection_fnc needs to be known at
    # jit time.
    @jax.jit
    def fnc(theta_tiles, is_null_per_arm, uniform_samples):
        sim_size, n_arm_samples, n_arms = uniform_samples.shape

        # 1. Calculate the binomial count data.
        # The sufficient statistic for binomial is just the number of uniform draws
        # above the threshold probability. But the `p_tiles` array has shape (n_tiles,
        # n_arms). So, we add empty dimensions to broadcast and then sum across
        # n_arm_samples to produce an output `y` array of shape: (n_tiles,
        # sim_size, n_arms)

        p_tiles = jax.scipy.special.expit(theta_tiles)
        y = jnp.sum(uniform_samples[None] < p_tiles[:, None, None, :], axis=2)

        # 2. Determine if we rejected each simulated sample.
        # rejection_fnc expects inputs of shape (n, n_arms) so we must flatten
        # our 3D arrays. We reshape exceedance afterwards to bring it back to 3D
        # (n_tiles, sim_size, n_arms)
        y_flat = y.reshape((-1, n_arms))
        n_flat = jnp.full_like(y_flat, n_arm_samples)
        data = jnp.stack((y_flat, n_flat), axis=-1)
        did_reject = rejection_fnc(data).reshape(y.shape)

        # 3. Determine type I family wise error rate.
        #  a. type I is only possible when the null hypothesis is true.
        #  b. check all null hypotheses.
        #  c. sum across all the simulations.
        false_reject = (
            did_reject
            & is_null_per_arm[
                :,
                None,
            ]
        )
        any_rejection = jnp.any(false_reject, axis=-1)
        typeI_sum = any_rejection.sum(axis=-1)

        # 4. Calculate score. The score function is the primary component of the
        #    gradient used in the bound:
        #  a. for binomial, it's just: y - n * p
        #  b. only summed when there is a rejection in the given simulation
        score = y - n_arm_samples * p_tiles[:, None, :]
        typeI_score = jnp.sum(any_rejection[:, :, None] * score, axis=1)

        return typeI_sum, typeI_score

    return fnc


def build_rejection_table(n_arms, n_arm_samples, rejection_fnc):
    """
    The Berry model generally deals with n_arm_samples <= 35. This means it is
    tractable to pre-calculate whether each dataset will reject the null because
    35^4 is a fairly manageable number. We can actually reduce the number of
    calculations because the arms are symmetric and we can run only for sorted
    datasets and then extrapolate to unsorted datasets.
    """

    # 1. Construct the n_arms-dimensional grid.
    ys = np.arange(n_arm_samples + 1)
    Ygrids = np.stack(np.meshgrid(*[ys] * n_arms, indexing="ij"), axis=-1)
    Yravel = Ygrids.reshape((-1, n_arms))

    # 2. Sort the grid arms while tracking the sorting order so that we can
    # unsort later.
    colsortidx = np.argsort(Yravel, axis=-1)
    inverse_colsortidx = np.zeros(Yravel.shape, dtype=np.int32)
    axis0 = np.arange(Yravel.shape[0])[:, None]
    inverse_colsortidx[axis0, colsortidx] = np.arange(n_arms)
    Y_colsorted = Yravel[axis0, colsortidx]

    # 3. Identify the unique datasets. In a 35^4 grid, this will be about 80k
    # datasets instead of 1.7m.
    Y_unique, inverse_unique = np.unique(Y_colsorted, axis=0, return_inverse=True)

    # 4. Compute the rejections for each unique dataset.
    N = np.full_like(Y_unique, n_arm_samples)
    data = np.stack((Y_unique, N), axis=-1)
    reject_unique = rejection_fnc(data)

    # 5. Invert the unique and the sort operations so that we know the rejection
    # value for every possible dataset.
    reject = reject_unique[inverse_unique][axis0, inverse_colsortidx]
    return reject


@jax.jit
def lookup_rejection(table, y, n_arm_samples=35):
    """
    Convert the y tuple datasets into indices and lookup from the table
    constructed by `build_rejection_table`.

    This assumes n_arm_samples is constant across arms.
    """
    n_arms = y.shape[-1]
    # Compute the strided array access. For example in 3D for y = [4,8,3], and
    # n_arm_samples=35, we'd have:
    # y_index = 4 * (36 ** 2) + 8 * (36 ** 1) + 3 * (36 ** 0)
    #         = 4 * (36 ** 2) + 8 * 36 + 3
    y_index = (y * ((n_arm_samples + 1) ** jnp.arange(n_arms)[::-1])[None, :]).sum(
        axis=-1
    )
    return table[y_index, :]


def upper_bound(
    theta_tiles,
    tile_radii,
    corners,
    sim_sizes,
    n_arm_samples,
    typeI_sum,
    typeI_score,
    delta=0.025,
    delta_prop_0to1=0.5,
):
    """
    Compute the Imprint upper bound after simulations have been run.
    """
    p_tiles = scipy.special.expit(theta_tiles)
    v_diff = corners - theta_tiles[:, None]
    v_sq = v_diff**2

    #
    # Step 1. 0th order terms.
    #
    # monte carlo estimate of type I error at each theta.
    d0 = typeI_sum / sim_sizes
    # clopper-pearson upper bound in beta form.
    d0u_factor = 1.0 - delta * delta_prop_0to1
    d0u = scipy.stats.beta.ppf(d0u_factor, typeI_sum + 1, sim_sizes - typeI_sum) - d0
    # If typeI_sum == sim_sizes, scipy.stats outputs nan. Output 0 instead
    # because there is no way to go higher than 1.0
    d0u = np.where(np.isnan(d0u), 0, d0u)

    #
    # Step 2. 1st order terms.
    #
    # Monte carlo estimate of gradient of type I error at the grid points
    # then dot product with the vector from the center to the corner.
    d1 = ((typeI_score / sim_sizes[:, None])[:, None] * v_diff).sum(axis=-1)
    d1u_factor = np.sqrt(1 / ((1 - delta_prop_0to1) * delta) - 1.0)
    covar_quadform = (
        n_arm_samples * v_sq * p_tiles[:, None] * (1 - p_tiles[:, None])
    ).sum(axis=-1)
    # Upper bound on d1!
    d1u = np.sqrt(covar_quadform) * (d1u_factor / np.sqrt(sim_sizes)[:, None])

    #
    # Step 3. 2nd order terms.
    #
    n_corners = corners.shape[1]

    p_lower = np.tile(
        scipy.special.expit(theta_tiles - tile_radii)[:, None], (1, n_corners, 1)
    )
    p_upper = np.tile(
        scipy.special.expit(theta_tiles + tile_radii)[:, None], (1, n_corners, 1)
    )

    special = (p_lower <= 0.5) & (0.5 <= p_upper)
    max_p = np.where(np.abs(p_upper - 0.5) < np.abs(p_lower - 0.5), p_upper, p_lower)
    hess_comp = np.where(special, 0.25 * v_sq, (max_p * (1 - max_p) * v_sq))

    hessian_quadform_bound = hess_comp.sum(axis=-1) * n_arm_samples
    d2u = 0.5 * hessian_quadform_bound

    #
    # Step 4. Identify the corners with the highest upper bound.
    #

    # The total of the bound component that varies between corners.
    total_var = d1u + d2u + d1
    total_var = np.where(np.isnan(total_var), 0, total_var)
    worst_corner = total_var.argmax(axis=1)
    ti = np.arange(d1.shape[0])
    d1w = d1[ti, worst_corner]
    d1uw = d1u[ti, worst_corner]
    d2uw = d2u[ti, worst_corner]

    #
    # Step 5. Compute the total bound and return it
    #
    total_bound = d0 + d0u + d1w + d1uw + d2uw

    return total_bound, d0, d0u, d1w, d1uw, d2uw
