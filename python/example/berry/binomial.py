import jax
import jax.numpy as jnp
import numpy as np


def binomial_accumulator(rejection_fnc):
    """
    A simple re-implementation of accumulation. This is useful for distilling
    what is happening during accumulation down to a simple linear sequence of
    operations. Retaining this could be useful for tutorials or conceptual
    introductions to Kevlar since we can explain this code without introducing
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
        did_reject = rejection_fnc(y_flat, n_flat).reshape(y.shape)

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
    ys = np.arange(n_arm_samples + 1)
    Ygrids = np.stack(np.meshgrid(*[ys] * n_arms, indexing="ij"), axis=-1)
    Yravel = Ygrids.reshape((-1, n_arms))

    colsortidx = np.argsort(Yravel, axis=-1)
    inverse_colsortidx = np.zeros(Yravel.shape, dtype=np.int32)
    axis0 = np.arange(Yravel.shape[0])[:, None]
    inverse_colsortidx[axis0, colsortidx] = np.arange(n_arms)
    Y_colsorted = Yravel[axis0, colsortidx]

    Y_unique, inverse_unique = np.unique(Y_colsorted, axis=0, return_inverse=True)

    N = np.full_like(Y_unique, n_arm_samples)
    reject_unique = rejection_fnc(Y_unique, N)
    reject = reject_unique[inverse_unique][axis0, inverse_colsortidx]
    return reject


@jax.jit
def lookup_rejection(table, y, n):
    y_index = (y * (36 ** jnp.arange(4)[::-1])[None, :]).sum(axis=-1)
    return table[y_index, :]


def cloudpickle_helper(fnc_pkl, *args):
    import cloudpickle

    return cloudpickle.loads(fnc_pkl)(*args)
