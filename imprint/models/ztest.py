import jax
import jax.numpy as jnp
import numpy as np

import imprint as ip


@jax.jit
def _sim(samples, theta, null_truth):
    return jnp.where(
        null_truth[:, None, 0],
        # negate so that we can do a less than comparison
        -(theta[:, None, 0] + samples[None, :]),
        jnp.inf,
    )


class ZTest1D(ip.Model):
    def __init__(self, seed, max_K, store=None):
        self.family = "normal"
        self.dtype = jnp.float32

        # key = jax.random.PRNGKey(seed)
        # self.samples = jax.random.normal(key, shape=(max_K,), dtype=self.dtype)

        np.random.seed(seed)
        self.samples = np.random.normal(size=(max_K,)).astype(self.dtype)

    def sim_batch(
        self,
        begin_sim: int,
        end_sim: int,
        theta: jnp.ndarray,
        null_truth: jnp.ndarray,
        detailed: bool = False,
    ):
        return _sim(self.samples[begin_sim:end_sim], theta, null_truth)
