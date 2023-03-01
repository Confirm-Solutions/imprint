import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.stats

import imprint as ip
import imprint.models.fisher_exact as fisher
from imprint.models.ztest import ZTest1D


def test_ztest_validate(snapshot):
    g = ip.cartesian_grid([-1], [1], n=[10], null_hypos=[ip.hypo("x < 0")])
    # lam = -1.96 because we negated the statistics so we can do a less than
    # comparison.
    lam = -1.96
    K = 2**13
    val_df = ip.validate(ZTest1D, g=g, lam=lam, K=K)

    # Check the TIE monte carlo is fairly close to the true error.
    # The limit is set to 1.2 because that is empirically the maximum that
    # results from the provided seed. This is not a rigorous test but it will
    # catch some types of errors.
    true_err = 1 - scipy.stats.norm.cdf(-g.get_theta()[:, 0] - lam)
    tie_est = val_df["tie_sum"] / K
    tie_std = scipy.stats.binom.std(n=K, p=true_err) / K
    n_stds = (tie_est - true_err) / tie_std
    assert np.all(np.abs(n_stds) < 1.2)

    # since forward(tie_cp_bound) == tie_bound, then
    # backward(tie_bound) == tie_cp_bound
    # check this!
    d = ip.driver.Driver(ZTest1D(0, K), tile_batch_size=1)
    theta, vertices = g.get_theta_and_vertices()
    f0 = d.backward_boundv(val_df["tie_bound"].to_numpy(), theta, vertices)
    np.testing.assert_allclose(f0, val_df["tie_cp_bound"])

    pd.testing.assert_frame_equal(val_df, snapshot(val_df))


def test_ztest_calibrate(snapshot):
    g = ip.cartesian_grid([-1], [1], n=[10], null_hypos=[ip.hypo("x < 0")])

    cal_df = ip.calibrate(ZTest1D, g=g)
    K = ip.driver.default_K
    np.testing.assert_allclose(np.floor((K + 1) * cal_df["alpha0"]) - 1, cal_df["idx"])

    # since backward(alpha) == alpha0, then forward(alpha0) == alpha
    # check this!
    theta, vertices = g.get_theta_and_vertices()
    d = ip.driver.Driver(ZTest1D(0, K), tile_batch_size=1)
    alpha = d.forward_boundv(cal_df["alpha0"].to_numpy(), theta, vertices)
    np.testing.assert_allclose(alpha, 0.025)

    pd.testing.assert_frame_equal(cal_df, snapshot(cal_df))


def test_jax_hypergeom():
    np.testing.assert_allclose(
        fisher.hypergeom_logpmf(3, 20, 10, 10),
        scipy.stats.hypergeom.logpmf(3, 20, 10, 10),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        fisher.hypergeom_logcdf(3, 20, 10, 10),
        scipy.stats.hypergeom.logcdf(3, 20, 10, 10),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        jnp.exp(fisher.hypergeom_logcdf(3, 20, 10, 10)),
        scipy.stats.hypergeom.cdf(3, 20, 10, 10),
        rtol=1e-5,
    )


def test_fisher_exact_jax_vs_scipy():
    model = fisher.FisherExact(0, 10, n=10)
    np.random.seed(0)
    theta = np.random.rand(5, 2)
    null_truth = np.ones((5, 1), dtype=bool)
    np.testing.assert_allclose(
        fisher._sim_scipy(model.samples[0:10], theta, null_truth),
        model.sim_batch(0, 10, theta, null_truth),
        rtol=1e-6,
    )
