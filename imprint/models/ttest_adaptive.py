import jax
import jax.numpy as jnp
import numpy as np


def _sim(C_avg, D_avg_sqrt, mu0, theta, null_truth, eff_size_thresh):
    sigma_sq = -0.5 / theta[:, 1]
    sigma = jnp.sqrt(sigma_sq)
    mu_div_sig = theta[:, 0] * sigma
    mu0_div_sig = mu0 / sigma
    shift = mu_div_sig - mu0_div_sig
    eff_size = C_avg[None] + shift[:, None, None]
    Ts = eff_size / D_avg_sqrt[None]
    Ts_subset = jnp.where(eff_size > eff_size_thresh, Ts, -jnp.inf)
    out = jnp.where(null_truth[:, None, 0], -jnp.max(Ts_subset, axis=-1), jnp.inf)
    return out


class TTest1DAda:
    def __init__(
        self,
        seed,
        max_K,
        n_init,
        n_samples_per_interim,
        n_interims,
        mu0,
        eff_size_thresh,
    ):
        n_samples_per_interim = np.array(n_samples_per_interim)
        if len(n_samples_per_interim.shape) == 0:
            n_samples_per_interim = np.full((n_interims,), n_samples_per_interim)
        elif (
            len(n_samples_per_interim.shape) > 1
            or n_samples_per_interim.shape[0] != n_interims
        ):
            raise Exception("n_samples_per_interim must be vector of length n_interims")

        self.n_samples_per_stage = np.concatenate([[n_init], n_samples_per_interim])
        self.N = np.cumsum(self.n_samples_per_stage)
        self.n_samples = self.N[-1]
        self.n_stages = len(self.N)

        self.family = "normal2"
        self.family_params = {"n": self.n_samples}
        self.dtype = jnp.float32
        self.mu0 = mu0
        self.n_interims = n_interims
        self.eff_size_thresh = eff_size_thresh

        key = jax.random.PRNGKey(seed)
        normals = jax.random.normal(key, shape=(max_K, self.n_stages))
        self.C = jnp.cumsum(jnp.sqrt(self.n_samples_per_stage)[None] * normals, axis=1)
        self.C_avg = self.C / self.N[None]

        _, key = jax.random.split(key)
        df = self.n_samples_per_stage - 1
        chisqs = 2 * jax.random.gamma(key, df / 2, shape=(max_K, self.n_stages))

        self.D = (self.n_samples_per_stage[1:] * self.N[:-1] / self.N[1:])[None] * (
            normals[:, 1:] / jnp.sqrt(self.n_samples_per_stage[None, 1:])
            - self.C_avg[:, :-1]
        ) ** 2
        self.D = jnp.concatenate([jnp.zeros(self.D.shape[0])[:, None], self.D], axis=1)
        self.D = jnp.cumsum(chisqs + self.D, axis=1)
        self.D_avg_sqrt = jnp.sqrt(self.D / (self.N[None] * (self.N[None] - 1)))

    def sim_batch(self, begin_sim, end_sim, theta, null_truth):
        return _sim(
            self.C_avg[begin_sim:end_sim],
            self.D_avg_sqrt[begin_sim:end_sim],
            self.mu0,
            theta,
            null_truth,
            self.eff_size_thresh,
        )
