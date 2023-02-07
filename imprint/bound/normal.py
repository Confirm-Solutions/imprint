"""
Normal Tilt-Bound with unknown mean and known variance (1 parameter).
"""
import jax
import jax.numpy as jnp


class ForwardQCPSolver:
    """
    Solves the minimization problem:
        0.5 * (q-1) * s_sq * v ** 2 - log(f0) / q
    with respect to q >= 1.
    """

    def __init__(self, scale):
        self.scale = scale

    def solve(self, v, f0):
        logf0 = jnp.log(f0)
        mv_sqrt = self.scale * jnp.abs(v)
        q_opt = jax.lax.cond(
            mv_sqrt == 0,
            lambda: jnp.inf,
            lambda: jnp.sqrt(-2 * logf0) / mv_sqrt,
        )
        return jnp.maximum(q_opt, 1)


class BackwardQCPSolver:
    """
    Solves the minimization problem:
        0.5 * q * s_sq * v ** 2 - log(alpha) * q / (q-1)
    with respect to q >= 1.
    """

    def __init__(self, scale):
        self.scale = scale

    def solve(self, v, alpha):
        mv_sqrt = self.scale * jnp.abs(v)
        return jax.lax.cond(
            mv_sqrt == 0,
            lambda: jnp.inf,
            lambda: 1 + jnp.sqrt(-2 * jnp.log(alpha)) / mv_sqrt,
        )


class TileForwardQCPSolver:
    """
    Solves the minimization problem:
        0.5 * (q-1) * s_sq * max_v v ** 2 - log(f0) / q
    with respect to q >= 1.
    """

    def __init__(self, scale):
        self.scale = scale

    def solve(self, vs, f0):
        logf0 = jnp.log(f0)
        mv_sqrt = self.scale * jnp.max(jnp.abs(vs))
        q_opt = jax.lax.cond(
            mv_sqrt == 0,
            lambda: jnp.inf,
            lambda: jnp.sqrt(-2 * logf0) / mv_sqrt,
        )
        return jnp.maximum(q_opt, 1)


class TileBackwardQCPSolver:
    """
    Solves the minimization problem:
        0.5 * q * s_sq * max_v v ** 2 - log(alpha) * q / (q-1)
    with respect to q >= 1.
    """

    def __init__(self, scale):
        self.scale = scale

    def solve(self, vs, alpha):
        mv_sqrt = self.scale * jnp.max(jnp.abs(vs))
        return jax.lax.cond(
            mv_sqrt == 0,
            lambda: jnp.inf,
            lambda: 1 + jnp.sqrt(-2 * jnp.log(alpha)) / mv_sqrt,
        )


def tilt_bound_fwd(q, scale, v, f0):
    p_inv = 1 - 1 / q
    expo = 0.5 * (q - 1) * (scale * v) ** 2
    return jax.lax.cond(
        (v == 0) | (scale == 0),
        lambda: f0**p_inv,
        lambda: jax.lax.cond(
            q == jnp.inf,
            lambda: jnp.inf,
            lambda: f0**p_inv * jnp.exp(expo),
        ),
    )


def tilt_bound_fwd_tile(q, scale, vs, f0):
    p_inv = 1 - 1 / q
    max_v = jnp.max(jnp.abs(vs))
    max_expo = 0.5 * (q - 1) * (scale * max_v) ** 2
    return jax.lax.cond(
        (max_v == 0) | (scale == 0),
        lambda: f0**p_inv,
        lambda: jax.lax.cond(
            q == jnp.inf,
            lambda: jnp.inf,
            lambda: f0**p_inv * jnp.exp(max_expo),
        ),
    )


def tilt_bound_bwd(q, scale, v, alpha):
    p = jax.lax.cond(q == 1, lambda _: jnp.inf, lambda q: 1 / (1 - 1 / q), q)
    expo = jax.lax.cond(
        (v == 0) | (scale == 0),
        lambda: 0.0,
        lambda: 0.5 * q * (scale * v) ** 2,
    )
    return alpha**p * jnp.exp(-expo)


def tilt_bound_bwd_tile(q, scale, vs, alpha):
    p = jax.lax.cond(q == 1, lambda _: jnp.inf, lambda q: 1 / (1 - 1 / q), q)
    max_v = jnp.max(jnp.abs(vs))
    max_expo = jax.lax.cond(
        (max_v == 0) | (scale == 0),
        lambda: 0.0,
        lambda: 0.5 * q * (scale * max_v) ** 2,
    )
    return alpha**p * jnp.exp(-max_expo)


class NormalBound:
    @staticmethod
    def get_backward_bound(family_params):
        scale = family_params.get("scale", 1.0)
        bwd_solver = TileBackwardQCPSolver(scale)

        def backward_bound(alpha_target, theta0, vertices):
            v = vertices - theta0
            q_opt = bwd_solver.solve(v, alpha_target)
            return tilt_bound_bwd_tile(q_opt, scale, v, alpha_target)

        return jax.jit(jax.vmap(backward_bound))

    @staticmethod
    def get_forward_bound(family_params):
        scale = family_params.get("scale", 1.0)
        fwd_solver = TileForwardQCPSolver(scale)

        def forward_bound(f0, theta0, vertices):
            vs = vertices - theta0
            q_opt = fwd_solver.solve(vs, f0)
            return tilt_bound_fwd_tile(q_opt, scale, vs, f0)

        return jax.jit(jax.vmap(forward_bound))
