import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.stats

import imprint as ip
import imprint.models.fisher_exact as fisher
from imprint.models.ztest import ZTest1D


def test_ztest(snapshot):
    g = ip.cartesian_grid([-1], [1], n=[10], null_hypos=[ip.hypo("x < 0")])
    # lam = -1.96 because we negated the statistics so we can do a less than
    # comparison.
    lam = -1.96
    K = 2**13
    rej_df = ip.validate(ZTest1D, g, lam, K=K)
    pd.testing.assert_frame_equal(rej_df, snapshot(rej_df))

    true_err = 1 - scipy.stats.norm.cdf(-g.get_theta()[:, 0] - lam)

    tie_est = rej_df["tie_sum"] / K
    tie_std = scipy.stats.binom.std(n=K, p=true_err) / K
    n_stds = (tie_est - true_err) / tie_std
    assert np.all(np.abs(n_stds) < 1.2)

    calibrate_df = ip.calibrate(ZTest1D, g)
    pd.testing.assert_frame_equal(calibrate_df, snapshot(calibrate_df))


def test_jax_hypergeom():
    np.testing.assert_allclose(
        fisher.hypergeom_logpmf(3, 20, 10, 10),
        scipy.stats.hypergeom.logpmf(3, 20, 10, 10),
    )
    np.testing.assert_allclose(
        fisher.hypergeom_logcdf(3, 20, 10, 10),
        scipy.stats.hypergeom.logcdf(3, 20, 10, 10),
    )
    np.testing.assert_allclose(
        jnp.exp(fisher.hypergeom_logcdf(3, 20, 10, 10)),
        scipy.stats.hypergeom.cdf(3, 20, 10, 10),
    )


def test_fisher_exact_jax_vs_scipy():
    model = fisher.FisherExact(0, 10, n=10)
    np.random.seed(0)
    theta = np.random.rand(5, 2)
    null_truth = np.ones((5, 1), dtype=bool)
    np.testing.assert_allclose(
        fisher._sim_scipy(model.samples[0:10], theta, null_truth),
        model.sim_batch(0, 10, theta, null_truth),
    )
