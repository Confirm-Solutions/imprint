import numpy as np
import jax
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist


def mcmc_berry(model, data, suc_thresh):
    logit_p1 = model.logit_p1

    def mcmc_berry_model(ys, ns):
        with numpyro.plate("i", 6, dim=-2):
            mu = numpyro.sample("mu", dist.Normal(-1.34, 10))
            sigma2 = numpyro.sample("sigma2", dist.InverseGamma(0.0005, 0.000005))
            with numpyro.plate("j", 4, dim=-1):
                theta = numpyro.sample(
                    "theta",
                    dist.Normal(mu, jax.numpy.sqrt(sigma2)),
                )
                numpyro.sample(
                    "y",
                    dist.BinomialLogits(theta + logit_p1, total_count=ns),
                    obs=ys,
                )

    mcmc_stats = {}
    mcmc_data = []
    nuts_kernel = NUTS(mcmc_berry_model)
    mcmc = MCMC(nuts_kernel, num_warmup=5000, num_samples=200_000, thinning=5, progress_bar=False)
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key, data[..., 0], data[..., 1])
    x = mcmc.get_samples(group_by_chain=True)
    summary = numpyro.diagnostics.summary(x, prob=0.95)
    mcmc_stats["cilow"] = summary["theta"]["2.5%"]
    mcmc_stats["cihi"] = summary["theta"]["97.5%"]
    mcmc_stats["theta_map"] = summary["theta"]["mean"]
    n_samples = x["theta"].shape[0] * x["theta"].shape[1]
    mcmc_stats["exceedance"] = (
        np.sum(x["theta"] > suc_thresh, axis=(0, 1)) / n_samples
    )
    mcmc_data.append(mcmc)
    return mcmc_data, mcmc_stats


if __name__ == "__main__":
    import berry
    from constants import DATA, DATA2

    b = berry.Berry(sigma2_n=90, sigma2_bounds=(1e-8, 1e3))
    mcmc_berry(b, DATA2, b.suc_thresh)
