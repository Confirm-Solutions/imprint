import jax
import numpy as np
import scipy as scipy

import imprint.bound.normal2 as normal2


def A(n, theta1, theta2):
    return np.sum(-0.25 * theta1**2 / theta2 - 0.5 * n * np.log(-2 * theta2))


def A_secant(n, theta1, theta2, v1, v2, q):
    return (A(n, theta1 + q * v1, theta2 + q * v2) - A(n, theta1, theta2)) / q


# only works for 1-sample, d-arms
def A_numerical(theta1, theta2):
    return np.sum(
        np.log(
            np.array(
                [
                    scipy.integrate.quad(
                        lambda x: np.exp(theta1[i] * x + theta2[i] * x**2),
                        -np.inf,
                        np.inf,
                    )[0]
                    for i in range(theta1.shape[0])
                ]
            )
        )
    )


def A_secant_numerical(theta1, theta2, v1, v2, q):
    return (
        A_numerical(theta1 + q * v1, theta2 + q * v2) - A_numerical(theta1, theta2)
    ) / q


def test_A_secant_unrestricted_no_nan():
    n = np.array([10, 10])
    theta1 = np.array([5, 2])
    theta2 = np.array([-0.1, -0.2])
    v1 = np.array([-0.2, 0.1])
    v2 = np.array([-0.1, -10])

    out = normal2.A_secant(n, theta1, theta2, v1, v2, q=1)
    assert not np.isnan(out)

    out = normal2.A_secant(n, theta1, theta2, v1, v2, q=5)
    assert not np.isnan(out)

    out = normal2.A_secant(n, theta1, theta2, v1, v2, q=1e30)
    assert not np.isnan(out)


def test_A_secant_coincide_A():
    n = np.array([10, 10])
    theta1 = np.array([5, 2])
    theta2 = np.array([-0.1, -0.2])
    v1 = np.array([-0.2, 0.1])
    v2 = np.array([-0.1, -10])
    qs = np.logspace(1, 3, 100)

    for q in qs:
        actual = normal2.A_secant(n, theta1, theta2, v1, v2, q)
        expected = A_secant(n, theta1, theta2, v1, v2, q)
        np.testing.assert_allclose(actual, expected)


def test_A_secant_coincide_A_numerical():
    n = np.array([1, 1])
    theta1 = np.array([5, 2])
    theta2 = np.array([-0.1, -0.2])
    v1 = np.array([-0.2, 0.1])
    v2 = np.array([-0.1, -10])
    qs = np.logspace(1, 3, 100)

    for q in qs:
        actual = normal2.A_secant(n, theta1, theta2, v1, v2, q)
        expected = A_secant_numerical(theta1, theta2, v1, v2, q)
        np.testing.assert_allclose(actual, expected)


def test_fwd_solver():
    n = np.array([10, 10])
    theta1 = np.array([5, 2])
    theta2 = np.array([-0.1, -0.2])
    v1 = np.array([-0.2, 0.1])
    v2 = np.array([-0.1, -0.1])
    alpha = 0.025
    q_dense = np.linspace(1, 1.1, 100000)

    solver = normal2.TileForwardQCPSolver(n, tol=1e-10, eps=1e-10)
    actual = solver.solve(theta1, theta2, v1[None], v2[None], alpha)
    expected = q_dense[
        np.argmin(
            jax.vmap(
                normal2.tilt_bound_fwd_tile,
                in_axes=(0, None, None, None, None, None, None),
            )(q_dense, n, theta1, theta2, v1[None], v2[None], alpha)
        )
    ]
    np.testing.assert_allclose(actual, expected)


def test_bwd_solver():
    n = np.array([10, 20])
    theta1 = np.array([5, 2])
    theta2 = np.array([-0.1, -0.2])
    v1 = np.array([-0.2, 0.1])
    v2 = np.array([0.05, -0.001])
    alpha = 0.025
    q_dense = np.linspace(1, 1.2, 100000)

    solver = normal2.TileBackwardQCPSolver(n, tol=1e-12, eps=1e-12)
    actual = solver.solve(theta1, theta2, v1[None], v2[None], alpha)
    expected = q_dense[
        np.argmax(
            jax.vmap(
                normal2.tilt_bound_bwd_tile,
                in_axes=(0, None, None, None, None, None, None),
            )(q_dense, n, theta1, theta2, v1[None], v2[None], alpha)
        )
    ]
    np.testing.assert_allclose(actual, expected)


def test_fwd_bound():
    n = np.array([10, 20])
    theta1 = np.array([5, 2])
    theta2 = np.array([-0.1, -0.2])
    v1 = np.array([-0.2, 0.1])
    v2 = np.array([0.05, -0.001])
    alpha = 0.025

    actual = normal2.tilt_bound_fwd_tile(
        np.inf, n, theta1, theta2, v1[None], v2[None], alpha
    )
    assert np.isnan(actual)

    v2 = np.array([-0.05, -0.001])
    actual = normal2.tilt_bound_fwd_tile(
        np.inf, n, theta1, theta2, v1[None], v2[None], alpha
    )
    assert not np.isnan(actual)


def test_bwd_bound():
    n = np.array([10, 20])
    theta1 = np.array([5, 2])
    theta2 = np.array([-0.1, -0.2])
    v1 = np.array([-0.2, 0.1])
    v2 = np.array([0.05, -0.001])
    alpha = 0.025

    actual = normal2.tilt_bound_bwd_tile(
        np.inf, n, theta1, theta2, v1[None], v2[None], alpha
    )
    assert np.isnan(actual)

    v2 = np.array([-0.05, -0.001])
    actual = normal2.tilt_bound_bwd_tile(
        np.inf, n, theta1, theta2, v1[None], v2[None], alpha
    )
    assert not np.isnan(actual)
