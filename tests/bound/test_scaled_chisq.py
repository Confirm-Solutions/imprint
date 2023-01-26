import jax
import numpy as np
import scipy as scipy

import imprint.bound.scaled_chisq as scaled_chisq


def A(n, df, theta):
    return -0.5 * np.sum(n * df * np.log(-2 * theta))


def A_secant(n, df, theta, v, q):
    return (A(n, df, theta + q * v) - A(n, df, theta)) / q


def A_numerical(df, theta):
    return np.sum(
        np.log(
            np.array(
                [
                    scipy.integrate.quad(
                        lambda x: x ** (df[i] / 2 - 1) * np.exp(theta[i] * x),
                        0,
                        np.inf,
                    )[0]
                    for i in range(theta.shape[0])
                ]
            )
        )
    )


def A_secant_numerical(df, theta, v, q):
    return (A_numerical(df, theta + q * v) - A_numerical(df, theta)) / q


def test_A_secant_unrestricted_no_nan():
    n = np.array([10, 10])
    df = np.array([1, 2])
    theta = np.array([-0.1, -0.2])
    v = np.array([-0.1, -10])

    out = scaled_chisq.A_secant(n, df, theta, v, q=1)
    assert not np.isnan(out)

    out = scaled_chisq.A_secant(n, df, theta, v, q=5)
    assert not np.isnan(out)

    out = scaled_chisq.A_secant(n, df, theta, v, q=1e30)
    assert not np.isnan(out)


def test_A_secant_coincide_A():
    n = np.array([10, 10])
    df = np.array([1, 20])
    theta = np.array([-0.1, -0.2])
    v = np.array([-0.1, -10])
    qs = np.logspace(1, 3, 100)

    for q in qs:
        actual = scaled_chisq.A_secant(n, df, theta, v, q)
        expected = A_secant(n, df, theta, v, q)
        np.testing.assert_allclose(actual, expected)


def test_A_secant_coincide_A_numerical():
    n = np.array([1, 1])
    df = np.array([3, 2])
    theta = np.array([-1, -2])
    v = np.array([-0.1, -0.2])
    qs = np.logspace(1, 3, 100)

    for q in qs:
        actual = scaled_chisq.A_secant(n, df, theta, v, q)
        expected = A_secant_numerical(df, theta, v, q)
        np.testing.assert_allclose(actual, expected)


def test_fwd_solver():
    n = np.array([10, 10])
    df = np.array([2, 20])
    theta = np.array([-0.1, -0.2])
    v = np.array([-0.1, -0.1])
    alpha = 0.025
    q_dense = np.linspace(1, 1.1, 100000)

    solver = scaled_chisq.TileForwardQCPSolver(n, df, tol=1e-10, eps=1e-10)
    actual = solver.solve(theta, v[None], alpha)
    expected = q_dense[
        np.argmin(
            jax.vmap(
                scaled_chisq.tilt_bound_fwd_tile,
                in_axes=(0, None, None, None, None, None),
            )(q_dense, n, df, theta, v[None], alpha)
        )
    ]
    np.testing.assert_allclose(actual, expected)


def test_bwd_solver():
    n = np.array([10, 10])
    df = np.array([2, 20])
    theta = np.array([-0.1, -0.2])
    v = np.array([-0.1, -0.1])
    alpha = 0.025
    q_dense = np.linspace(1.7, 1.9, 100000)

    solver = scaled_chisq.TileBackwardQCPSolver(n, df, tol=1e-10, eps=1e-10)
    actual = solver.solve(theta, v[None], alpha)
    expected = q_dense[
        np.argmax(
            jax.vmap(
                scaled_chisq.tilt_bound_bwd_tile,
                in_axes=(0, None, None, None, None, None),
            )(q_dense, n, df, theta, v[None], alpha)
        )
    ]
    np.testing.assert_allclose(actual, expected)


def test_fwd_bound():
    n = np.array([10, 20])
    df = np.array([5, 2])
    theta = np.array([-0.1, -0.2])
    v = np.array([0.05, -0.001])
    alpha = 0.025

    actual = scaled_chisq.tilt_bound_fwd_tile(np.inf, n, df, theta, v[None], alpha)
    assert np.isnan(actual)

    v = np.array([-0.05, -0.001])
    actual = scaled_chisq.tilt_bound_fwd_tile(np.inf, n, df, theta, v[None], alpha)
    assert not np.isnan(actual)


def test_bwd_bound():
    n = np.array([10, 20])
    df = np.array([5, 2])
    theta = np.array([-0.1, -0.2])
    v = np.array([0.05, -0.001])
    alpha = 0.025

    actual = scaled_chisq.tilt_bound_bwd_tile(np.inf, n, df, theta, v[None], alpha)
    assert np.isnan(actual)

    v = np.array([-0.05, -0.001])
    actual = scaled_chisq.tilt_bound_bwd_tile(np.inf, n, df, theta, v[None], alpha)
    assert not np.isnan(actual)
