import numpy as np
import jax
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist


def mcmc_berry(model, data, suc_thresh):
    logit_p1 = model.logit_p1

    def mcmc_berry_model(y, n):
        mu = numpyro.sample("mu", dist.Normal(-1.34, 10))
        sigma2 = numpyro.sample("sigma2", dist.InverseGamma(0.0005, 0.000005))
        theta = numpyro.sample(
            "theta",
            dist.MultivariateNormal(
                mu, jax.numpy.diag(jax.numpy.repeat(jax.numpy.sqrt(sigma2), 4))
            ),
        )
        with numpyro.plate("i", 4):
            numpyro.sample(
                "y",
                dist.BinomialLogits(theta + logit_p1, total_count=n),
                obs=y,
            )

    mcmc_stats = dict(
        cilow=np.empty((6, 4)),
        cihi=np.empty((6, 4)),
        theta_map=np.empty((6, 4)),
        exceedance=np.empty((6, 4)),
    )
    mcmc_data = []
    for i in range(data.shape[0]):
        y = data[i, ..., 0]
        n = data[i, ..., 1]
        nuts_kernel = NUTS(mcmc_berry_model)
        mcmc = MCMC(
            nuts_kernel, num_warmup=5000, num_samples=20000, thinning=5
        )
        rng_key = jax.random.PRNGKey(0)
        mcmc.run(
            rng_key,
            y,
            n,
        )
        x = mcmc.get_samples(group_by_chain=True)
        summary = numpyro.diagnostics.summary(x, prob=0.95)
        mcmc_stats["cilow"][i] = summary["theta"]["2.5%"]
        mcmc_stats["cihi"][i] = summary["theta"]["97.5%"]
        mcmc_stats["theta_map"][i] = summary["theta"]["mean"]
        n_samples = x["theta"].shape[0] * x["theta"].shape[1]
        mcmc_stats["exceedance"][i] = (
            np.sum(x["theta"] > suc_thresh[i], axis=(0, 1)) / n_samples
        )
        mcmc_data.append(mcmc)
    return mcmc_data, mcmc_stats
