import jax
import jax.numpy as jnp


def _simple_bisection(f, m, M, tol):
    def _cond_fun(args):
        m, M = args
        return (M - m) / M > tol

    def _body_fun(args):
        m, M = args
        x = jnp.linspace(m, M, 4)
        y = f(x)
        i_star = jnp.argmin(y)
        new_min = jnp.where(
            i_star <= 1,
            m,
            x[i_star - 1],
        )
        new_max = jnp.where(
            i_star <= 1,
            x[i_star + 1],
            M,
        )
        return (
            new_min,
            new_max,
        )

    _init_val = (m, M)
    m, M = jax.lax.while_loop(
        _cond_fun,
        _body_fun,
        _init_val,
    )
    return (M + m) / 2.0
