import jax
import jax.numpy as jnp
import numpy as np
from berrylib.fast_math import (
    jax_fast_invert,
    jax_faster_inv,
    jax_faster_inv_diag,
    jax_faster_inv_product,
    jax_faster_log_det,
    log_normal_pdf,
)


def test_faster_linalg():
    num_iter = 1000
    key = jax.random.PRNGKey(0)
    d = jax.random.uniform(key, (num_iter, 4))
    s = jax.random.uniform(key, (num_iter, 1))
    g = jax.random.uniform(key, (num_iter, 4))
    atol = 1e-6
    for d, s, g in zip(d, s, g):
        m = np.diag(d) + s
        sign, expected = np.linalg.slogdet(m)
        if sign < 0:
            # If the determinant is negative, the matrix is not positive definite
            continue
        # Determinant
        np.testing.assert_allclose(jax_faster_log_det(d, s), expected, atol=atol)

        # Inverse
        expected = np.linalg.inv(m)
        np.testing.assert_allclose(jax_faster_inv(d, s), expected, atol=atol)
        np.testing.assert_allclose(
            jax_fast_invert(
                jnp.linalg.inv(jnp.full_like(expected, s) + jnp.diag(jnp.repeat(1, 4))),
                d - 1,
            ),
            expected,
            atol=atol,
        )

        # Inverse product
        expected = np.linalg.inv(m) @ g
        np.testing.assert_allclose(jax_faster_inv_product(d, s, g), expected, atol=atol)

        # Inverse diagonal
        expected = np.diag(np.linalg.inv(m))
        np.testing.assert_allclose(jax_faster_inv_diag(d, s), expected, atol=atol)


def test_log_normal_pdf():
    x = np.array([0.0, 0.5])
    mu = np.array([0, 1])
    cov = np.array([[2, 0.5], [0.5, 2]])
    prec = np.linalg.inv(cov)
    vals, vecs = np.linalg.eigh(prec)
    expected = jax.scipy.stats.multivariate_normal.logpdf(x, mu, cov)
    np.testing.assert_allclose(
        log_normal_pdf(x, mu, vals, vecs, omit_constants=False), expected
    )
