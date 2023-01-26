import jax
import jax.numpy as jnp


@jax.jit
def _sim(samples, theta, null_truth):
    return jnp.where(
        null_truth[:, None, 0],
        0.5 * samples[None, :] / theta[:, None, 1],
        jnp.inf,
    )


class ChiSqTest:
    def __init__(self, seed, max_K, n_samples, family="normal2", store=None):
        df = n_samples - 1

        self.family = family
        if family == "normal2":
            self.family_params = {"n": n_samples}
        elif family == "scaled_chisq":
            self.family_params = {"n": 1, "df": df}
        else:
            raise RuntimeError("Unrecognized family.")
        self.dtype = jnp.float32

        # sample n_samples number of N(0, 1) and compute variance estimator.
        # This is equivalent to directly sampling chi^2(n-1) ~ Gamma((n-1)/2, 2)
        key = jax.random.PRNGKey(seed)
        self.samples = 2 * jax.random.gamma(
            key, df / 2, shape=(max_K,), dtype=self.dtype
        )

    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        return _sim(self.samples[begin_sim:end_sim], theta, null_truth)
