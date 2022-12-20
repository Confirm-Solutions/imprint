import jax
import jax.numpy as jnp
import numpy as np
import scipy


# We reimplement the hypergeometric PDF and CDF in JAX for performance.
def hypergeom_logpmf(k, M, n, N):
    # Copied from scipy.stats.hypergeom
    tot, good = M, n
    bad = tot - good
    betaln = jax.scipy.special.betaln
    result = (
        betaln(good + 1, 1)
        + betaln(bad + 1, 1)
        + betaln(tot - N + 1, N + 1)
        - betaln(k + 1, good - k + 1)
        - betaln(N - k + 1, bad - N + k + 1)
        - betaln(tot + 1, 1)
    )
    return result


def hypergeom_logcdf(k, M, n, N):
    return jax.lax.fori_loop(
        1,
        k + 1,
        lambda i, acc: jax.scipy.special.logsumexp(
            jnp.array([acc, hypergeom_logpmf(i, M, n, N)])
        ),
        hypergeom_logpmf(0, M, n, N),
    )


def hypergeom_cdf(k, M, n, N):
    return jnp.exp(hypergeom_logcdf(k, M, n, N))


def scipy_fisher_exact(tbl):
    return scipy.stats.fisher_exact(tbl, alternative="less")[1]


def _sim_scipy(samples, theta, null_truth, f=None):
    if f is None:
        f = scipy_fisher_exact

    p = scipy.special.expit(theta)
    successes = np.sum(samples[None, :] < p[:, None, None], axis=2)
    tbl2by2 = np.concatenate(
        (successes[:, :, None, :], samples.shape[1] - successes[:, :, None, :]),
        axis=2,
    )
    stats = np.array(
        [
            [f(tbl2by2[i, j]) for j in range(tbl2by2.shape[1])]
            for i in range(tbl2by2.shape[0])
        ]
    )
    return stats


@jax.jit
def _sim_jax(samples, theta, null_truth):
    n = samples.shape[1]
    p = jax.scipy.special.expit(theta)
    successes = jnp.sum(samples[None, :] < p[:, None, None], axis=2)
    cdfvv = jax.vmap(
        jax.vmap(hypergeom_cdf, in_axes=(0, None, 0, None)),
        in_axes=(0, None, 0, None),
    )
    cdf = cdfvv(successes[..., 0], 2 * n, successes.sum(axis=-1), n)
    return cdf


class FisherExact:
    def __init__(self, seed, max_K, *, n):
        self.family = "binomial"
        self.family_params = {"n": n}
        self.samples = jax.random.uniform(
            jax.random.PRNGKey(seed), shape=(max_K, n, 2), dtype=jnp.float32
        )

    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        return _sim_jax(self.samples[begin_sim:end_sim], theta, null_truth)


class BoschlooExact(FisherExact):
    # NOTE: This is super slow!
    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        def f(tbl):
            return scipy.stats.boschloo_exact(tbl, alternative="less").pvalue

        return _sim_scipy(self.samples[begin_sim:end_sim], theta, null_truth, f=f)


class BarnardExact(FisherExact):
    # NOTE: This is super slow!
    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        def f(tbl):
            return scipy.stats.barnard_exact(tbl, alternative="less").pvalue

        return _sim_scipy(self.samples[begin_sim:end_sim], theta, null_truth, f=f)
