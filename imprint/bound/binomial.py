import jax
import jax.numpy as jnp

from . import optimizer as opt


def logistic(t):
    """
    Numerically stable implementation of log(1 + e^t).
    """
    return jnp.maximum(t, 0) + jnp.log(1 + jnp.exp(-jnp.abs(t)))


def logistic_secant(t, v, q, b):
    """
    Numerically stable implementation of the secant of logistic defined by:
        (logistic(t + q * v) - logistic(b)) / q
    defined for all t, v in R and q > 0.
    It is only numerically stable if t, b are not too large in magnitude
    and q is sufficiently away from 0.
    """
    t_div_q = t / q
    ls_1 = jnp.maximum(t_div_q + v, 0) - jnp.maximum(b, 0) / q
    ls_2 = jnp.log(1 + jnp.exp(-jnp.abs(t + q * v)))
    ls_2 = ls_2 - jnp.log(1 + jnp.exp(-jnp.abs(b)))
    ls_2 = ls_2 / q
    return ls_1 + ls_2


def A_secant(n, t, v, q, b):
    """
    Numerically stable implementation of the secant of A:
        (A(t + q * v) - A(b)) / q
    """
    return jnp.sum(n * logistic_secant(t, v, q, b))


class BaseTileQCPSolver:
    def __init__(self, n, m=1, M=1e7, tol=1e-5):
        self.n = n
        self.min = m
        self.max = M
        self.tol = tol


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
            theta_0,
        )
        sec1 = A_secant(
            self.n,
            theta_0,
            v,
            1,
            theta_0,
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
        return jax.lax.cond(
            loga < -1e10,
            lambda: jnp.inf,
            lambda: opt._simple_bisection(
                lambda x: self.obj_vmap(theta_0, vs, x, loga),
                self.min,
                self.max,
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
            theta_0,
        )
        sec1 = A_secant(
            self.n,
            theta_0,
            v,
            1,
            theta_0,
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
        return jax.lax.cond(
            loga < -1e10,
            lambda: jnp.inf,
            lambda: opt._simple_bisection(
                lambda x: self.obj_vmap(theta_0, vs, x, loga),
                self.min,
                self.max,
                self.tol,
            ),
        )


def tilt_bound_fwd(
    q,
    n,
    theta_0,
    v,
    f0,
):
    """
    Computes the forward q-Holder bound given by:
        f0 * exp[L(q) - (A(theta_0 + v) - A(theta_0))]
    for fixed f0, n, theta_0, v,
    where L, A are as given in ForwardQCPSolver.

    Parameters:
    -----------
    q:      q parameter.
    n:      scalar Binomial size parameter.
    theta_0:    d-array pivot point.
    v:          d-array displacement vector.
    f0:         probability value at theta_0.
    """
    expo = A_secant(n, theta_0, v, q, theta_0)
    expo = expo - A_secant(n, theta_0, v, 1, theta_0)
    return f0 ** (1 - 1 / q) * jnp.exp(expo)


def tilt_bound_fwd_tile(
    q,
    n,
    theta_0,
    vs,
    f0,
):
    """
    Computes the forward q-Holder bound given by:
        f0 * max_{v in vs} exp[L(q) - (A(theta_0 + v) - A(theta_0))]
    for fixed f0, n, theta_0, vs,
    where L, A are as given in ForwardQCPSolver.

    Parameters:
    -----------
    q:      q parameter.
    n:      scalar Binomial size parameter.
    theta_0:    d-array pivot point.
    vs:         (k, d)-array of displacement vectors
                denoting the corners of a rectangle.
    f0:         probability value at theta_0.
    """

    def _expo(v):
        expo = A_secant(n, theta_0, v, q, theta_0)
        expo = expo - A_secant(n, theta_0, v, 1, theta_0)
        return expo

    max_expo = jnp.max(jax.vmap(_expo, in_axes=(0,))(vs))
    return f0 ** (1 - 1 / q) * jnp.exp(max_expo)


def tilt_bound_bwd(
    q,
    n,
    theta_0,
    v,
    alpha,
):
    """
    Computes the backward q-Holder bound given by:
        exp(-L(q))
    where L(q) is as given in BackwardQCPSolver.
    The resulting value is alpha' such that
        q_holder_bound_fwd(q, n, theta_0, v, alpha') = alpha

    Parameters:
    -----------
    q:      q parameter.
    n:      scalar Binomial size parameter.
    theta_0:    d-array pivot point.
    v:          d-array displacement from pivot point.
    alpha:      target level.
    """

    def _bound(q):
        p = 1 / (1 - 1 / q)
        slope_diff = A_secant(n, theta_0, v, q, theta_0)
        slope_diff = slope_diff - A_secant(n, theta_0, v, 1, theta_0)
        return (alpha * jnp.exp(-slope_diff)) ** p

    return jax.lax.cond(
        q <= 1,
        lambda _: (alpha >= 1) + 0.0,
        _bound,
        q,
    )


def tilt_bound_bwd_tile(
    q,
    n,
    theta_0,
    vs,
    alpha,
):
    """
    Computes the backward q-Holder bound given by:
        max_{v in vs} exp(-L(q))
    where L(q) is as given in BackwardQCPSolver.

    Parameters:
    -----------
    q:      q parameter.
    n:      scalar Binomial size parameter.
    theta_0:    d-array pivot point.
    vs:         (k, d)-array displacement from pivot point.
                These represent the corners of the rectangular tile.
    alpha:      target level.
    """
    p = 1 / (1 - 1 / q)

    def _expo(v):
        slope_diff = A_secant(n, theta_0, v, q, theta_0)
        slope_diff = slope_diff - A_secant(n, theta_0, v, 1, theta_0)
        return slope_diff

    def _bound():
        max_expo = jnp.max(jax.vmap(_expo, in_axes=(0,))(vs))
        return (alpha * jnp.exp(-max_expo)) ** p

    return jax.lax.cond(
        q <= 1,
        lambda: (alpha >= 1) + 0.0,
        _bound,
    )


class BinomialBound:
    @staticmethod
    def get_backward_bound(family_params):
        n = family_params["n"]
        bwd_solver = TileBackwardQCPSolver(n)

        def backward_bound(alpha_target, theta0, vertices):
            v = vertices - theta0
            q_opt = bwd_solver.solve(theta0, v, alpha_target)
            return tilt_bound_bwd_tile(q_opt, n, theta0, v, alpha_target)

        return jax.jit(jax.vmap(backward_bound, in_axes=(None, 0, 0)))

    @staticmethod
    def get_forward_bound(family_params):
        n = jnp.asarray(family_params["n"])
        fwd_solver = TileForwardQCPSolver(n)

        def forward_bound(f0, theta0, vertices):
            vs = vertices - theta0
            q_opt = fwd_solver.solve(theta0, vs, f0)
            return tilt_bound_fwd_tile(q_opt, n, theta0, vs, f0)

        return jax.jit(jax.vmap(forward_bound))
