import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


@jax.jit
def _sim(samples, theta, null_truth):
    p = jax.scipy.special.expit(theta)
    stats = jnp.sum(samples[None, :] < p[:, None], axis=2) / samples.shape[1]
    return jnp.where(
        null_truth[:, None, 0],
        1 - stats,
        jnp.inf,
    )


def unifs(seed, *, shape, dtype):
    samples = jax.random.uniform(
        jax.random.PRNGKey(seed), shape=shape, dtype=dtype
    ).ravel()
    return pd.DataFrame(dict(data=[samples.tobytes()]))


class Binom1D:
    def __init__(self, seed, max_K, *, n, store=lambda x: x):
        self.family = "binomial"
        self.family_params = {"n": n}
        self.dtype = jnp.float32

        samples_bytes = store(unifs)(seed, shape=(max_K, n), dtype=self.dtype)
        # NOTE: reshape before converting to jax because jax copies on reshape
        self.samples = jnp.asarray(
            np.frombuffer(samples_bytes["data"].iloc[0], self.dtype).reshape((max_K, n))
        )

    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        return _sim(self.samples[begin_sim:end_sim], theta, null_truth)
