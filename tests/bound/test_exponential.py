import jax
import numpy as np

import imprint.bound.exponential as exponential
import imprint.bound.scaled_chisq as scaled_chisq


def A(n, theta):
    return -np.sum(n * np.log(-theta))


def A_secant(n, theta, v, q):
    return (A(n, theta + q * v) - A(n, theta)) / q


def A_numerical(theta):
    import scipy.integrate

    return np.sum(
        np.log(
            np.array(
                [
                    scipy.integrate.quad(
                        lambda x: np.exp(theta[i] * x),
                        0,
                        np.inf,
                    )[0]
                    for i in range(theta.shape[0])
                ]
            )
        )
    )


def A_secant_numerical(theta, v, q):
    return (A_numerical(theta + q * v) - A_numerical(theta)) / q


def test_A_secant_unrestricted_no_nan():
    n = np.array([10, 10])
    theta = np.array([-0.1, -0.2])
    v = np.array([-0.1, -10])

    out = exponential.A_secant(n, theta, v, q=1)
    assert not np.isnan(out)

    out = exponential.A_secant(n, theta, v, q=5)
    assert not np.isnan(out)

    out = exponential.A_secant(n, theta, v, q=1e30)
    assert not np.isnan(out)


def test_A_secant_coincide_A():
    n = np.array([10, 10])
    theta = np.array([-0.1, -0.2])
    v = np.array([-0.1, -10])
    qs = np.logspace(1, 3, 100)

    for q in qs:
        actual = exponential.A_secant(n, theta, v, q)
        expected = A_secant(n, theta, v, q)
        np.testing.assert_allclose(actual, expected)


def test_A_secant_coincide_A_numerical():
    n = np.array([1, 1])
    theta = np.array([-1, -2])
    v = np.array([-0.1, -0.2])
    qs = np.logspace(1, 3, 100)

    for q in qs:
        actual = exponential.A_secant(n, theta, v, q)
        expected = A_secant_numerical(theta, v, q)
        np.testing.assert_allclose(actual, expected)


def test_A_secant_coincide_scaled_chisq():
    n = np.array([10, 10])
    df = np.array([2, 2])  # for equivalence
    theta = np.array([-1, -2])
    v = np.array([-0.1, -0.2])
    qs = np.logspace(1, 3, 100)

    for q in qs:
        actual = exponential.A_secant(n, theta, v, q)
        expected = scaled_chisq.A_secant(n, df, theta, v, q)
        np.testing.assert_allclose(actual, expected)


def test_fwd_solver():
    n = np.array([10, 10])
    theta = np.array([-0.1, -0.2])
    v = np.array([-0.1, -0.1])
    alpha = 0.025
    q_dense = np.linspace(1.27, 1.29, 100000)

    solver = exponential.TileForwardQCPSolver(n, tol=1e-10, eps=1e-10)
    actual = solver.solve(theta, v[None], alpha)
    expected = q_dense[
        np.argmin(
            jax.vmap(
                exponential.tilt_bound_fwd_tile,
                in_axes=(0, None, None, None, None),
            )(q_dense, n, theta, v[None], alpha)
        )
    ]
    np.testing.assert_allclose(actual, expected)


def test_bwd_solver():
    n = np.array([10, 10])
    theta = np.array([-0.1, -0.2])
    v = np.array([-0.1, -0.1])
    alpha = 0.025
    q_dense = np.linspace(3.2, 3.4, 100000)

    solver = exponential.TileBackwardQCPSolver(n, tol=1e-10, eps=1e-10)
    actual = solver.solve(theta, v[None], alpha)
    expected = q_dense[
        np.argmax(
            jax.vmap(
                exponential.tilt_bound_bwd_tile,
                in_axes=(0, None, None, None, None),
            )(q_dense, n, theta, v[None], alpha)
        )
    ]
    np.testing.assert_allclose(actual, expected)


def test_fwd_bound():
    n = np.array([10, 20])
    theta = np.array([-0.1, -0.2])
    v = np.array([0.05, -0.001])
    alpha = 0.025

    actual = exponential.tilt_bound_fwd_tile(np.inf, n, theta, v[None], alpha)
    assert np.isnan(actual)

    v = np.array([-0.05, -0.001])
    actual = exponential.tilt_bound_fwd_tile(np.inf, n, theta, v[None], alpha)
    assert not np.isnan(actual)


def test_bwd_bound():
    n = np.array([10, 20])
    theta = np.array([-0.1, -0.2])
    v = np.array([0.05, -0.001])
    alpha = 0.025

    actual = exponential.tilt_bound_bwd_tile(np.inf, n, theta, v[None], alpha)
    assert np.isnan(actual)

    v = np.array([-0.05, -0.001])
    actual = exponential.tilt_bound_bwd_tile(np.inf, n, theta, v[None], alpha)
    assert not np.isnan(actual)
