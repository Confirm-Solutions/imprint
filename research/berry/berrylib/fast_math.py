from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=["omit_constants"])
def log_normal_pdf(x, mean, prec_eig_vals, prec_eig_vecs, omit_constants=True):
    """Compute the log of the multivariate normal pdf.

    Using the eigendecomposition is numerically stable. Adapted from scipy.
    """
    logdet = -jnp.sum(jnp.log(prec_eig_vals))
    U = prec_eig_vecs * jnp.sqrt(prec_eig_vals)
    dev = x - mean
    # "maha" for "Mahalanobis distance".
    maha = jnp.square(jnp.dot(dev, U)).sum()
    if omit_constants:
        return -0.5 * (maha + logdet)
    else:
        rank = len(prec_eig_vals)
        log2pi = jnp.log(2 * jnp.pi)
        return -0.5 * (rank * log2pi + maha + logdet)


@jax.jit
def jax_fast_invert(S, d):
    """
    Invert a matrix plus a diagonal by iteratively applying the Sherman-Morrison
    formula. If we are computing Binv = (A + d)^-1,
    then the arguments are:
    - S: A^-1
    - d: d
    """
    # NOTE: It's possible to improve performance by about 10% by doing an
    # incomplete inversion here. In the last iteration through the loop, return
    # both S and offset. Then, perform .dot(grad) with those components directly.
    for k in range(d.shape[0]):
        offset = d[k] / (1 + d[k] * S[k, k])
        S = S - (offset * (S[k, None, :] * S[:, None, k]))
    return S


@jax.jit
def jax_faster_inv(D, S):
    """Compute the inverse of a diagonal matrix D plus a shift S.

    This function uses "Sherman-Morrison" formula:
    https://en.wikipedia.org/wiki/Sherman–Morrison_formula
    """
    D_inverse = 1.0 / D
    # NB: reusing D_inverse in this line is numerically unstable
    multiplier = -S / (1 + (S / D).sum())
    M = multiplier * jnp.outer(D_inverse, D_inverse)
    M = M + jnp.diag(D_inverse)
    return M


@jax.jit
def jax_faster_inv_diag(D, S):
    """Compute the diagonal of the inverse of a diagonal matrix D plus a shift S.

    This function uses "Sherman-Morrison" formula:
    https://en.wikipedia.org/wiki/Sherman–Morrison_formula
    """
    D_inverse = 1.0 / D
    # NB: reusing D_inverse in this line is numerically unstable
    multiplier = -S / (1 + (S / D).sum())
    return multiplier * D_inverse * D_inverse + D_inverse


@jax.jit
def jax_faster_inv_product(D, S, G):
    """Compute (diag(D)+S)^-1 @ G.

    This function uses "Sherman-Morrison" formula:
    https://en.wikipedia.org/wiki/Sherman–Morrison_formula
    """
    D_norm = jnp.abs(D).sum()
    D_normed = D / D_norm
    return (-S * (G / D_normed).sum() / (D_norm + (S / D_normed).sum()) + G) / D


@jax.jit
def jax_faster_log_det(D, S):
    """Compute the log determinant of a diagnal matrix D plus a shift S.

    Valid only if the determinant is positive.

    This function uses "Sherman-Morrison for determinants"
    https://en.wikipedia.org/wiki/Matrix_determinant_lemma
    """
    detD_inverse = jnp.log(D).sum()
    newdeterminant = detD_inverse + jnp.log1p((S / D).sum())
    return newdeterminant
