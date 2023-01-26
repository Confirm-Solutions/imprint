"""
Exponential Tilt-Bound with unknown scale parameter.
Assumes multi-arm with possibly different
(scale,) parameters in each arm as well as sample size.
"""
import jax
import jax.numpy as jnp

from . import optimizer as opt


def A_secant(n, theta, v, q):
    """
    Computes the secant of the log-partition function:
        (A(theta + q * v) - A(theta)) / q

    Parameters:
    -----------
    n:  a scalar or (d,)-vector of sample sizes for each arm.
    theta:      a (d,)-array where theta1[i] is the
                natural parameter of arm i.
    v:          a (d,)-array of displacements.
    q:          secant slope.
    """
    return -jnp.sum(n * jnp.log(1 + q * v / theta)) / q


class BaseTileQCPSolver:
    def __init__(self, n, m=1, M=1e7, tol=1e-5, eps=1e-6):
        self.n = n
        self.min = m
        self.max = M
        self.tol = tol
        self.shrink_factor = 1 - eps

    def _compute_max_bound(self, theta_0, vs):
        max_vs = jnp.max(vs, axis=0)

        # return shrunken maximum so that the
        # maximum q results in well-defined objective.
        return jnp.minimum(
            self.max,
            jnp.min(
                jnp.where(
                    max_vs > 0,
                    -theta_0 / max_vs * self.shrink_factor,
                    jnp.inf,
                )
            ),
        )


class TileForwardQCPSolver(BaseTileQCPSolver):
    r"""
    Solves the following strictly quasi-convex optimization problem:
        minimize_q max_{v \in S} L_v(q)
        subject to q >= 1
    where
        L_v(q) = (psi(theta_0, v, q) - log(a)) / q - psi(theta_0, v, 1)
    """

    def obj_v(self, theta_0, v, q, loga):
        secq = A_secant(
            self.n,
            theta_0,
            v,
            q,
        )
        sec1 = A_secant(
            self.n,
            theta_0,
            v,
            1,
        )
        return secq - loga / q - sec1

    def obj(self, theta_0, vs, q, loga):
        _obj_each_vmap = jax.vmap(self.obj_v, in_axes=(None, 0, None, None))
        return jnp.max(_obj_each_vmap(theta_0, vs, q, loga))

    def obj_vmap(self, theta_0, vs, qs, loga):
        return jax.vmap(
            self.obj,
            in_axes=(None, None, 0, None),
        )(theta_0, vs, qs, loga)

    def solve(self, theta_0, vs, a):
        loga = jnp.log(a)
        max_trunc = self._compute_max_bound(theta_0, vs)
        return jax.lax.cond(
            loga < -1e10,
            lambda: jnp.inf,
            lambda: opt._simple_bisection(
                lambda x: self.obj_vmap(theta_0, vs, x, loga),
                self.min,
                max_trunc,
                self.tol,
            ),
        )


class TileBackwardQCPSolver(BaseTileQCPSolver):
    r"""
    Solves the following strictly quasi-convex optimization problem:
        minimize_q max_{v \in S} L_v(q)
        subject to q >= 1
    where
        L_v(q) = (q/(q-1)) * (psi(theta_0, v, q) / q - psi(theta_0, v, 1) - log(a))
    """

    def obj_v(self, theta_0, v, q):
        secq = A_secant(
            self.n,
            theta_0,
            v,
            q,
        )
        sec1 = A_secant(
            self.n,
            theta_0,
            v,
            1,
        )
        return secq - sec1

    def obj(self, theta_0, vs, q, loga):
        p = 1.0 / (1.0 - 1.0 / q)
        _obj_each_vmap = jax.vmap(self.obj_v, in_axes=(None, 0, None))
        return p * (jnp.max(_obj_each_vmap(theta_0, vs, q)) - loga)

    def obj_vmap(self, theta_0, vs, qs, loga):
        return jax.vmap(
            self.obj,
            in_axes=(None, None, 0, None),
        )(theta_0, vs, qs, loga)

    def solve(self, theta_0, vs, a):
        loga = jnp.log(a)
        max_trunc = self._compute_max_bound(theta_0, vs)
        return jax.lax.cond(
            loga < -1e10,
            lambda: jnp.inf,
            lambda: opt._simple_bisection(
                lambda x: self.obj_vmap(theta_0, vs, x, loga),
                self.min,
                max_trunc,
                self.tol,
            ),
        )


def tilt_bound_fwd_tile(
    q,
    n,
    theta_0,
    vs,
    f0,
):
    def _expo(v):
        expo = A_secant(n, theta_0, v, q)
        expo = expo - A_secant(n, theta_0, v, 1)
        return expo

    def _expo_inf(v):
        expo = jnp.where(jnp.any(v > 0), jnp.nan, 0)
        expo = expo - A_secant(n, theta_0, v, 1)
        return expo

    return jax.lax.cond(
        q == jnp.inf,
        lambda: f0 * jnp.exp(jnp.max(jax.vmap(_expo_inf, in_axes=(0,))(vs))),
        lambda: f0 ** (1 - 1 / q) * jnp.exp(jnp.max(jax.vmap(_expo, in_axes=(0,))(vs))),
    )


def tilt_bound_bwd_tile(
    q,
    n,
    theta_0,
    vs,
    alpha,
):
    p = 1 / (1 - 1 / q)

    def _expo(v):
        slope_diff = A_secant(n, theta_0, v, q)
        slope_diff = slope_diff - A_secant(n, theta_0, v, 1)
        return slope_diff

    def _expo_inf(v):
        expo = jnp.where(jnp.any(v > 0), jnp.nan, 0)
        expo = expo - A_secant(n, theta_0, v, 1)
        return expo

    def _bound():
        max_expo = jnp.max(jax.vmap(_expo, in_axes=(0,))(vs))
        return (alpha * jnp.exp(-max_expo)) ** p

    def _bound_inf():
        max_expo = jnp.max(jax.vmap(_expo_inf, in_axes=(0,))(vs))
        return (alpha * jnp.exp(-max_expo)) ** p

    return jax.lax.cond(
        q <= 1,
        lambda: (alpha >= 1) + 0.0,
        lambda: jax.lax.cond(
            q == jnp.inf,
            _bound_inf,
            _bound,
        ),
    )


class ExponentialBound:
    @staticmethod
    def get_backward_bound(family_params):
        n = family_params["n"]
        bwd_solver = TileBackwardQCPSolver(n)

        def backward_bound(alpha_target, theta0, vertices):
            vs = vertices - theta0
            q_opt = bwd_solver.solve(theta0, vs, alpha_target)
            return tilt_bound_bwd_tile(q_opt, n, theta0, vs, alpha_target)

        return jax.jit(jax.vmap(backward_bound, in_axes=(None, 0, 0)))

    @staticmethod
    def get_forward_bound(family_params):
        n = family_params["n"]
        fwd_solver = TileForwardQCPSolver(n)

        def forward_bound(f0, theta0, vertices):
            vs = vertices - theta0
            q_opt = fwd_solver.solve(theta0, vs, f0)
            return tilt_bound_fwd_tile(q_opt, n, theta0, vs, f0)

        return jax.jit(jax.vmap(forward_bound))
