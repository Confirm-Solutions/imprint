# flake8: noqa E402
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=6"
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.infer

# This line is critical for enabling 64-bit floats.
from jax.config import config

config.update("jax_enable_x64", True)


@profile
def mcmc_berry(data, logit_p1, suc_thresh, n_arms=4, dtype=np.float64, n_samples=10000):
    @profile
    def mcmc_berry_model(y, n):
        mu = numpyro.sample("mu", dist.Normal(-1.34, 10))

        # An attempt at sampling from the logarithm to make the conditioning better.
        # LogTransform = dist.transforms._InverseTransform(dist.transforms.ExpTransform())
        # log_inverse_gamma = dist.TransformedDistribution(
        #     dist.InverseGamma(0.0005, 0.000005),
        #     LogTransform
        # )
        # log_sigma2 = numpyro.sample("log_sigma2", log_inverse_gamma)
        # sigma2 = numpyro.deterministic("sigma2", jnp.exp(log_sigma2))

        sigma2 = numpyro.sample("sigma2", dist.InverseGamma(0.0005, 0.000005))
        with numpyro.plate("j", n_arms):
            theta = numpyro.sample(
                "theta",
                dist.Normal(mu, jax.numpy.sqrt(sigma2)),
            )
            numpyro.sample(
                "y",
                dist.BinomialLogits(theta + logit_p1, total_count=n),
                obs=y,
            )

    # Number of devices to pmap over
    n_data = data.shape[0]
    rng_keys = jax.random.split(jax.random.PRNGKey(0), n_data)

    # NOTE: pmap requires an exact match between data.shape[0] and the number of
    # devices. so to use it further with more than n_cores datasets, we would
    # need to batch the data and layer a vmap underneath
    # traces = jax.pmap(do_mcmc)(rng_keys, data[..., 0], data[..., 1])

    mcmc_stats = dict(
        cilow=np.empty((n_data, n_arms)),
        cihi=np.empty((n_data, n_arms)),
        theta_map=np.empty((n_data, n_arms)),
        exceedance=np.empty((n_data, n_arms)),
        x=[None] * n_data,
        summary=[None] * n_data,
    )
    nuts_kwargs = dict(step_size=0.002, adapt_step_size=False)
    nuts_kernel = numpyro.infer.NUTS(mcmc_berry_model, **nuts_kwargs)
    mcmc = numpyro.infer.MCMC(
        nuts_kernel,
        progress_bar=False,
        num_warmup=1000,
        num_samples=n_samples,
    )
    for i in range(n_data):
        y = jnp.asarray(data[i, :, 0], dtype=dtype)
        n = jnp.asarray(data[i, :, 1], dtype=dtype)
        mcmc.run(rng_keys[i], y, n)
        x = mcmc.get_samples(group_by_chain=True)
        assert x["theta"].dtype == dtype
        summary = numpyro.diagnostics.summary(x, prob=0.95)
        mcmc_stats["cilow"][i] = summary["theta"]["2.5%"]
        mcmc_stats["cihi"][i] = summary["theta"]["97.5%"]
        mcmc_stats["theta_map"][i] = summary["theta"]["mean"]
        n_samples = x["theta"].shape[0] * x["theta"].shape[1]
        mcmc_stats["exceedance"][i] = (
            np.sum(x["theta"] > suc_thresh[i], axis=(0, 1)) / n_samples
        )
        mcmc_stats["x"][i] = x
        mcmc_stats["summary"][i] = summary
    return mcmc_stats


if __name__ == "__main__":
    import berry
    from constants import DATA2

    b = berry.Berry(sigma2_n=90, sigma2_bounds=(1e-6, 1e3), n_arms=2)
    N = 2
    n_i = np.full((N, 2), 35)
    y_i = np.full_like(n_i, 5)
    data = np.stack((y_i, n_i), axis=-1)
    results = mcmc_berry(data, b.logit_p1, np.full(N, b.suc_thresh[0, 0]), n_arms=2)
