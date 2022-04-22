# flake8: noqa E402
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=6"
import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def mcmc_berry(model, data, suc_thresh):
    logit_p1 = model.logit_p1

    def mcmc_berry_model(y, n):
        mu = numpyro.sample("mu", dist.Normal(-1.34, 10))
        sigma2 = numpyro.sample("sigma2", dist.InverseGamma(0.0005, 0.000005))
        with numpyro.plate("j", 4):
            theta = numpyro.sample(
                "theta",
                dist.Normal(mu, jax.numpy.sqrt(sigma2)),
            )
            numpyro.sample(
                "y",
                dist.BinomialLogits(theta + logit_p1, total_count=n),
                obs=y,
            )

    def do_mcmc(rng_key, y, n):
        nuts_kernel = NUTS(mcmc_berry_model)
        mcmc = MCMC(
            nuts_kernel,
            progress_bar=False,
            num_warmup=10_000,
            num_samples=500_000,
            thinning=5,
        )
        mcmc.run(rng_key, y, n)
        return mcmc.get_samples(group_by_chain=True)

    # Number of devices to pmap over
    n_parallel = data.shape[0]
    rng_keys = jax.random.split(jax.random.PRNGKey(0), n_parallel)
    traces = jax.pmap(do_mcmc)(rng_keys, data[..., 0], data[..., 1])
    # concatenate traces along pmap'ed axis

    mcmc_stats = dict(
        cilow=np.empty((6, 4)),
        cihi=np.empty((6, 4)),
        theta_map=np.empty((6, 4)),
        exceedance=np.empty((6, 4)),
    )
    assert data.shape[0] == 6
    for i in range(6):
        x = {k: v[i] for k, v in traces.items()}
        summary = numpyro.diagnostics.summary(x, prob=0.95)
        mcmc_stats["cilow"][i] = summary["theta"]["2.5%"]
        mcmc_stats["cihi"][i] = summary["theta"]["97.5%"]
        mcmc_stats["theta_map"][i] = summary["theta"]["mean"]
        n_samples = x["theta"].shape[0] * x["theta"].shape[1]
        mcmc_stats["exceedance"][i] = (
            np.sum(x["theta"] > suc_thresh[i], axis=(0, 1)) / n_samples
        )
        return mcmc_stats


if __name__ == "__main__":
    import berry
    from constants import DATA2

    b = berry.Berry(sigma2_n=90, sigma2_bounds=(1e-8, 1e3))
    mcmc_berry(b, DATA2, b.suc_thresh)
