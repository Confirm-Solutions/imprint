# flake8: noqa E402
import os

# TODO: this could be set more globally and in a flexible way, but for now, this seems fine.
# I think we should have a jax_setup module that sets variables like this before
# importing jax.
n_requested_cores_mcmc = 8
os.environ[
    "XLA_FLAGS"
] = f"--xla_force_host_platform_device_count={n_requested_cores_mcmc}"
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.infer

# This line is critical for enabling 64-bit floats.
from jax.config import config

config.update("jax_enable_x64", True)


def mcmc_berry(
    data, logit_p1, suc_thresh, dtype=np.float64, n_samples=10000, sigma2_val=None
):
    n_arms = data.shape[-2]

    def mcmc_berry_model(y, n):
        mu = numpyro.sample("mu", dist.Normal(-1.34, 10))

        # NOTE: An attempt at sampling from the logarithm to make the
        # conditioning better and avoid the need for small step_size. This
        # didn't work but maybe something like it would work, so I left the code
        # here.
        # LogTransform = dist.transforms._InverseTransform(dist.transforms.ExpTransform())
        # log_inverse_gamma = dist.TransformedDistribution(
        #     dist.InverseGamma(0.0005, 0.000005),
        #     LogTransform
        # )
        # log_sigma2 = numpyro.sample("log_sigma2", log_inverse_gamma)
        # sigma2 = numpyro.deterministic("sigma2", jnp.exp(log_sigma2))

        if sigma2_val is None:
            sigma2 = numpyro.sample("sigma2", dist.InverseGamma(0.0005, 0.000005))
        else:
            sigma2 = sigma2_val
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
    seed = 0

    def do_mcmc(rng_key, y, n):
        # Small step_size is necessary for the sampler to notice the density lying
        # in [1e-6, 1e-3]. Without this, results are very wrong.
        if sigma2_val is None:
            nuts_kwargs = dict(step_size=0.001, adapt_step_size=False)
        else:
            nuts_kwargs = dict()
        nuts_kernel = numpyro.infer.NUTS(mcmc_berry_model, **nuts_kwargs)
        mcmc = numpyro.infer.MCMC(
            nuts_kernel,
            progress_bar=False,
            num_warmup=1000,
            num_samples=n_samples,
        )
        mcmc.run(rng_key, y, n)
        return mcmc.get_samples(group_by_chain=True)

    p_do_mcmc = jax.pmap(do_mcmc)

    mcmc_stats = dict(
        cilow=np.empty((n_data, n_arms)),
        cihi=np.empty((n_data, n_arms)),
        theta_map=np.empty((n_data, n_arms)),
        exceedance=np.empty((n_data, n_arms)),
        x=[None] * n_data,
        summary=[None] * n_data,
    )
    n_cores_mcmc = jax.local_device_count()
    for i in range(0, n_data, n_cores_mcmc):
        chunk_end = i + n_cores_mcmc
        data_chunk = data[i:chunk_end]

        # This is ugly but kind of fun. Just pad the array to have length
        # n_cores_mcmc in the first dimension if the chunk is smaller the
        # n_cores_mcmc
        if chunk_end > n_data:
            data_chunk = np.pad(
                data_chunk, [(0, chunk_end - n_data), (0, 0), (0, 0)], mode="symmetric"
            )

        # We need to generate new keys each time because they are immutable.
        # Otherwise, the same random numbers will be re-used over and over.
        rng_keys = jax.random.split(jax.random.PRNGKey(seed + i), n_cores_mcmc)
        traces = p_do_mcmc(rng_keys, data_chunk[..., 0], data_chunk[..., 1])

        for j in range(min(chunk_end, n_data) - i):
            x = {k: v[j] for k, v in traces.items()}
            summary = numpyro.diagnostics.summary(x, prob=0.95)
            mcmc_stats["cilow"][i + j] = summary["theta"]["2.5%"]
            mcmc_stats["cihi"][i + j] = summary["theta"]["97.5%"]
            mcmc_stats["theta_map"][i + j] = summary["theta"]["mean"]
            n_samples = x["theta"].shape[0] * x["theta"].shape[1]
            mcmc_stats["exceedance"][i + j] = (
                np.sum(x["theta"] > suc_thresh[i + j], axis=(0, 1)) / n_samples
            )
            mcmc_stats["x"][i + j] = x
            mcmc_stats["summary"][i + j] = summary
    return mcmc_stats


def calc_pdf(x, bin_midpts, bin_wts):
    bin_edges = np.empty(bin_midpts.shape[0] + 1)
    bin_edges[1:-1] = (bin_midpts[:-1] + bin_midpts[1:]) * 0.5
    bin_edges[0] = bin_midpts[0] - (bin_midpts[1] - bin_midpts[0]) / 2
    bin_edges[-1] = bin_midpts[-1] - (bin_midpts[-2] - bin_midpts[-1]) / 2

    pdf, _ = np.histogram(x, bin_edges)
    pdf = pdf.astype(np.float64)
    bin_width = bin_edges[1:] - bin_edges[:-1]
    pdf /= bin_width
    normalize = (bin_wts * pdf).sum()
    return pdf / normalize
