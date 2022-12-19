import jax
import jax.numpy as jnp


def _quad_form(v, A):
    return v.dot(A @ v)


class ForwardQCPSolver:
    """
    Solves the minimization problem:
        0.5 * (q-1) * v^T cov v - log(f0) / q
    with respect to q >= 1.
    """

    def __init__(self, cov):
        self.cov = cov

    def solve(self, v, f0):
        logf0 = jnp.log(f0)
        mv = _quad_form(v, self.cov)
        q_opt = jnp.sqrt(-2 * logf0 / mv)
        return jnp.maximum(q_opt, 1)


class BackwardQCPSolver:
    """
    Solves the minimization problem:
        0.5 * q * v^T cov v - log(alpha) * q / (q-1)
    with respect to q >= 1.
    """

    def __init__(self, cov):
        self.cov = cov

    def solve(self, v, alpha):
        mv = _quad_form(v, self.cov)
        return 1 + jnp.sqrt(-2 * jnp.log(alpha) / mv)


class TileForwardQCPSolver:
    """
    Solves the minimization problem:
        0.5 * (q-1) * max_v v^T cov v - log(f0) / q
    with respect to q >= 1.
    """

    def __init__(self, cov):
        self.cov = cov

    def solve(self, vs, f0):
        logf0 = jnp.log(f0)
        mv = jnp.max(jax.vmap(_quad_form, in_axes=(0, None))(vs, self.cov))
        q_opt = jnp.sqrt(-2 * logf0 / mv)
        return jnp.maximum(q_opt, 1)


class TileBackwardQCPSolver:
    """
    Solves the minimization problem:
        0.5 * q * max_v v^T cov v - log(alpha) * q / (q-1)
    with respect to q >= 1.
    """

    def __init__(self, cov):
        self.cov = cov

    def solve(self, vs, alpha):
        mv = jnp.max(jax.vmap(_quad_form, in_axes=(0, None))(vs, self.cov))
        return 1 + jnp.sqrt(-2 * jnp.log(alpha) / mv)


def tilt_bound_fwd(q, cov, v, f0):
    p_inv = 1 - 1 / q
    expo = 0.5 * (q - 1) * _quad_form(v, cov)
    return f0**p_inv * jnp.exp(expo)


def tilt_bound_fwd_tile(q, cov, vs, f0):
    def _compute_expo(v):
        return 0.5 * (q - 1) * _quad_form(v, cov)

    p_inv = 1 - 1 / q
    max_expo = jnp.max(jax.vmap(_compute_expo, in_axes=(0,))(vs))
    return f0**p_inv * jnp.exp(max_expo)


def tilt_bound_bwd(q, cov, v, alpha):
    p = 1 / (1 - 1 / q)
    expo = 0.5 * (q - 1) * _quad_form(v, cov)
    return (alpha * jnp.exp(-expo)) ** p


def tilt_bound_bwd_tile(q, cov, vs, alpha):
    def _compute_expo(v):
        return 0.5 * (q - 1) * _quad_form(v, cov)

    p = 1 / (1 - 1 / q)
    max_expo = jnp.max(jax.vmap(_compute_expo, in_axes=(0,))(vs))
    return (alpha * jnp.exp(-max_expo)) ** p
