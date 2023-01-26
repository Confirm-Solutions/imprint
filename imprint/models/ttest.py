import jax
import jax.numpy as jnp


@jax.jit
def _sim(normals, sqrt_chisqs_scaled, mu0, sqrt_n_samples, theta, null_truth):
    sigma = jnp.sqrt(-0.5 / theta[:, 1])
    mu_div_sig = theta[:, 0] * sigma
    shift = sqrt_n_samples * (mu_div_sig - mu0 / sigma)
    Ts = (normals[None, :] + shift[:, None]) / sqrt_chisqs_scaled[None, :]
    return jnp.where(
        null_truth[:, None, 0],
        -Ts,
        jnp.inf,
    )


class TTest1D:
    def __init__(self, seed, max_K, n_samples, mu0):
        self.family = "normal2"
        self.family_params = {"n": n_samples}
        self.dtype = jnp.float32
        self.mu0 = mu0
        self.sqrt_n_samples = jnp.sqrt(n_samples)

        key = jax.random.PRNGKey(seed)
        self.normals = jax.random.normal(key, shape=(max_K,), dtype=self.dtype)
        _, key = jax.random.split(key)
        df = n_samples - 1
        chisqs = 2 * jax.random.gamma(key, df / 2, shape=(max_K,), dtype=self.dtype)
        self.sqrt_chisqs_scaled = jnp.sqrt(chisqs / (n_samples - 1))

    def sim_batch(self, begin_sim, end_sim, theta, null_truth):
        return _sim(
            self.normals[begin_sim:end_sim],
            self.sqrt_chisqs_scaled[begin_sim:end_sim],
            self.mu0,
            self.sqrt_n_samples,
            theta,
            null_truth,
        )
