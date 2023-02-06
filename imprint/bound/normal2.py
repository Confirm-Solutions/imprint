"""
Normal Tilt-Bound with unknown mean and variance (2 parameters).
Assumes multi-arm Normal with possibly different
(mean, variance) parameters and sample size in each arm.

The natural parameters for the Normal distribution are:
theta_0 = mu / sigma^2
theta_1 = -1 / (2 * sigma^2)
"""
import jax
import jax.numpy as jnp

from . import optimizer as opt


def A_secant(n, theta1, theta2, v1, v2, q):
    """
    Numerically stable implementation of the secant of A:
        (A(theta + qv) - A(theta)) / q
    It is only well-defined if q < min_{i : v2[i] > 0} (v2[i] / theta2[i])
    (if the set is empty, then the minimum is defined to be infinity).
    No error-checking is done for performance reasons.

    Parameters:
    ----------
    n:  a scalar or (d,)-vector of sample sizes for each arm.
    theta1:     a (d,)-array where theta1[i] is the
                1st order natural parameter of arm i.
    theta2:     a (d,)-array where theta2[i] is the
                2nd order natural parameter of arm i.
    v1:         a (d,)-array displacement on theta1.
    v2:         a (d,)-array displacement on theta2.
    q:          tilt parameter.
    """
    return (
        -0.25
        * jnp.sum(
            (theta2 * v1 * (q * v1 + 2 * theta1) - theta1**2 * v2)
            / (theta2 * (theta2 + q * v2))
        )
        - 0.5 * jnp.sum(n * jnp.log(1 + q * v2 / theta2)) / q
    )


class BaseTileQCPSolver:
    def __init__(self, n, m=1, M=1e7, tol=1e-5, eps=1e-6):
        self.n = n
        self.min = m
        self.max = M
        self.tol = tol
        self.shrink_factor = 1 - eps

    def _compute_max_bound(self, theta_02, v2s):
        max_v2s = jnp.max(v2s, axis=0)

        # return shrunken maximum so that the
        # maximum q results in well-defined objective.
        return jnp.maximum(
            self.min,
            jnp.min(
                jnp.where(
                    max_v2s > 0,
                    -theta_02 / max_v2s * self.shrink_factor,
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

    def obj_v(self, theta_01, theta_02, v1, v2, q, loga):
        secq = A_secant(
            self.n,
            theta_01,
            theta_02,
            v1,
            v2,
            q,
        )
        sec1 = A_secant(
            self.n,
            theta_01,
            theta_02,
            v1,
            v2,
            1,
        )
        return secq - loga / q - sec1

    def obj(self, theta_01, theta_02, v1s, v2s, q, loga):
        _obj_each_vmap = jax.vmap(self.obj_v, in_axes=(None, None, 0, 0, None, None))
        return jnp.max(_obj_each_vmap(theta_01, theta_02, v1s, v2s, q, loga))

    def obj_vmap(self, theta_01, theta_02, v1s, v2s, qs, loga):
        return jax.vmap(
            self.obj,
            in_axes=(None, None, None, None, 0, None),
        )(theta_01, theta_02, v1s, v2s, qs, loga)

    def solve(self, theta_01, theta_02, v1s, v2s, a):
        loga = jnp.log(a)
        max_q_valid = self._compute_max_bound(theta_02, v2s)
        return jax.lax.cond(
            loga < -1e10,
            lambda: max_q_valid,
            lambda: opt._simple_bisection(
                lambda x: self.obj_vmap(theta_01, theta_02, v1s, v2s, x, loga),
                self.min,
                jnp.minimum(max_q_valid, self.max),
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

    def obj_v(self, theta_01, theta_02, v1, v2, q):
        secq = A_secant(
            self.n,
            theta_01,
            theta_02,
            v1,
            v2,
            q,
        )
        sec1 = A_secant(
            self.n,
            theta_01,
            theta_02,
            v1,
            v2,
            1,
        )
        return secq - sec1

    def obj(self, theta_01, theta_02, v1s, v2s, q, loga):
        p = 1.0 / (1.0 - 1.0 / q)
        _obj_each_vmap = jax.vmap(self.obj_v, in_axes=(None, None, 0, 0, None))
        return p * (jnp.max(_obj_each_vmap(theta_01, theta_02, v1s, v2s, q)) - loga)

    def obj_vmap(self, theta_01, theta_02, v1s, v2s, qs, loga):
        return jax.vmap(
            self.obj,
            in_axes=(None, None, None, None, 0, None),
        )(theta_01, theta_02, v1s, v2s, qs, loga)

    def solve(self, theta_01, theta_02, v1s, v2s, a):
        loga = jnp.log(a)
        max_q_valid = self._compute_max_bound(theta_02, v2s)
        return jax.lax.cond(
            loga < -1e10,
            lambda: max_q_valid,
            lambda: opt._simple_bisection(
                lambda x: self.obj_vmap(theta_01, theta_02, v1s, v2s, x, loga),
                self.min,
                jnp.minimum(max_q_valid, self.max),
                self.tol,
            ),
        )


def tilt_bound_fwd_tile(
    q,
    n,
    theta_01,
    theta_02,
    v1s,
    v2s,
    f0,
):
    def _expo(v1, v2):
        expo = A_secant(n, theta_01, theta_02, v1, v2, q)
        expo = expo - A_secant(n, theta_01, theta_02, v1, v2, 1)
        return expo

    def _expo_inf(v1, v2):
        expo = jnp.sum(v1**2 / v2) + jnp.where(jnp.any(v2 > 0), jnp.nan, 0)
        expo = expo - A_secant(n, theta_01, theta_02, v1, v2, 1)
        return expo

    return jax.lax.cond(
        q == jnp.inf,
        lambda: f0 * jnp.exp(jnp.max(jax.vmap(_expo_inf, in_axes=(0, 0))(v1s, v2s))),
        lambda: f0 ** (1 - 1 / q)
        * jnp.exp(jnp.max(jax.vmap(_expo, in_axes=(0, 0))(v1s, v2s))),
    )


def tilt_bound_bwd_tile(
    q,
    n,
    theta_01,
    theta_02,
    v1s,
    v2s,
    alpha,
):
    p = 1 / (1 - 1 / q)

    def _expo(v1, v2):
        slope_diff = A_secant(n, theta_01, theta_02, v1, v2, q)
        slope_diff = slope_diff - A_secant(n, theta_01, theta_02, v1, v2, 1)
        return slope_diff

    def _expo_inf(v1, v2):
        expo = jnp.sum(v1**2 / v2) + jnp.where(jnp.any(v2 > 0), jnp.nan, 0)
        expo = expo - A_secant(n, theta_01, theta_02, v1, v2, 1)
        return expo

    def _bound():
        max_expo = jnp.max(jax.vmap(_expo, in_axes=(0, 0))(v1s, v2s))
        return (alpha * jnp.exp(-max_expo)) ** p

    def _bound_exp():
        max_expo = jnp.max(jax.vmap(_expo_inf, in_axes=(0, 0))(v1s, v2s))
        return (alpha * jnp.exp(-max_expo)) ** p

    return jax.lax.cond(
        q <= 1,
        lambda: (alpha >= 1) + 0.0,
        lambda: jax.lax.cond(
            q == jnp.inf,
            _bound_exp,
            _bound,
        ),
    )


class Normal2Bound:
    @staticmethod
    def get_backward_bound(family_params):
        n = family_params["n"]
        bwd_solver = TileBackwardQCPSolver(n)

        def backward_bound(alpha_target, theta0, vertices):
            v = vertices - theta0
            mid = len(theta0) // 2
            theta01, theta02 = theta0[:mid], theta0[mid:]
            v1s, v2s = v[:, 0], v[:, 1]
            q_opt = bwd_solver.solve(theta01, theta02, v1s, v2s, alpha_target)
            return tilt_bound_bwd_tile(
                q_opt, n, theta01, theta02, v1s, v2s, alpha_target
            )

        jit_bwd = jax.jit(jax.vmap(backward_bound, in_axes=(None, 0, 0)))

        def f(alpha_target, theta0, vertices):
            if jnp.any(vertices[..., 1] >= 0):
                raise ValueError(
                    "theta[1] must be negative for normal2."
                    " The natural parameters are (mu/sigma^2, -1/(2*sigma^2))."
                )
            return jit_bwd(alpha_target, theta0, vertices)

        return f

    @staticmethod
    def get_forward_bound(family_params):
        n = family_params["n"]
        fwd_solver = TileForwardQCPSolver(n)

        def forward_bound(f0, theta0, vertices):
            v = vertices - theta0
            mid = len(theta0) // 2
            theta01, theta02 = theta0[:mid], theta0[mid:]
            v1s, v2s = v[:, 0], v[:, 1]
            q_opt = fwd_solver.solve(theta01, theta02, v1s, v2s, f0)
            return tilt_bound_fwd_tile(q_opt, n, theta01, theta02, v1s, v2s, f0)

        jit_fwd = jax.jit(jax.vmap(forward_bound))

        def f(f0, theta0, vertices):
            if jnp.any(vertices[..., 1] >= 0):
                raise ValueError(
                    "theta[1] must be negative for normal2."
                    " The natural parameters are (mu/sigma^2, -1/(2*sigma^2))."
                )
            return jit_fwd(f0, theta0, vertices)

        return f
