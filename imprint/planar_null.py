import logging
import warnings
from dataclasses import dataclass

import numpy as np
import sympy as sp

from . import grid

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class HyperPlane(grid.NullHypothesis):
    """
    A hyperplane defined by:
    x.dot(n) - c = 0

    Sign convention: When used as the boundary between null hypothesis and
    alternative, the normal should point towards the null hypothesis space.
    """

    n: np.ndarray
    c: float

    def __post_init__(self):
        self.n = np.asarray(self.n)
        self.c = float(self.c)

    def __eq__(self, other):
        if not isinstance(other, HyperPlane):
            return NotImplemented
        return np.allclose(self.n, other.n) and np.isclose(self.c, other.c)

    def _pad_n(self, d):
        hp_dim = self.n.shape[0]

        if d < hp_dim:
            raise ValueError(
                f"HyperPlane has higher dimension (d={hp_dim})" f" than grid (d={d})."
            )
        elif d > hp_dim:
            logger.debug(
                "HyperPlane has dimension %s but grid"
                " has dimension %s. Padding with zeros.",
                hp_dim,
                d,
            )
            return np.pad(self.n, (0, d - hp_dim))
        else:
            return self.n

    def use_fast_path(self):
        return True

    def dist(self, theta):
        n = self._pad_n(theta.shape[-1])
        return theta.dot(n) - self.c

    def side(self, g):
        _, vertices = g.get_theta_and_vertices()
        eps = 1e-15
        vertex_dist = self.dist(vertices)
        side = np.zeros(vertices.shape[0], dtype=np.int8)
        side[(vertex_dist >= -eps).all(axis=-1)] = 1
        side[(vertex_dist <= eps).all(axis=-1)] = -1
        return side, vertex_dist[side == 0]

    def split(self, g, vertex_dist):
        eps = 1e-15
        n = self._pad_n(g.d)
        theta, vertices = g.get_theta_and_vertices()
        radii = g.get_radii()

        ########################################
        # Step 1. Intersect tile edges with the hyperplane.
        # This will identify the new vertices that we need to add.
        ########################################
        split_edges = grid.get_edges(theta, radii)
        # The first n_params columns of split_edges are the vertices from which
        # the edge originates and the second n_params are the edge vector.
        split_vs = split_edges[..., : g.d]
        split_dir = split_edges[..., g.d :]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Intersect each edge with the plane.
            alpha = (self.c - split_vs.dot(n)) / (split_dir.dot(n))
            # Now we need to identify the new tile vertices. We have three
            # possible cases here:
            # 1. Intersection: indicated by 0 < alpha < 1. We give a little
            #    eps slack to ignore intersections for null planes that just barely
            #    touch a corner of a tile. In this case, we
            # 2. Non-intersection indicated by alpha not in [0, 1]. In this
            #    case, the new vertex will just be marked nan to be filtered out
            #    later.
            # 3. Non-finite alpha which also indicates no intersection. Again,
            #    we produced a nan vertex to filter out later.
            new_vs = split_vs + alpha[:, :, None] * split_dir
            new_vs = np.where(
                (np.isfinite(new_vs)) & ((alpha > eps) & (alpha < 1 - eps))[..., None],
                new_vs,
                np.nan,
            )

        ########################################
        # Step 2. Construct the vertex array for the new tiles..
        ########################################
        # Create the array for the new vertices. We need to expand the
        # original vertex array in both dimensions:
        # 1. We create a new row for each tile that is being split using np.repeat.
        # 2. We create a new column for each potential additional vertex from
        #    the intersection operation above using np.concatenate. This is
        #    more new vertices than necessary, but facilitates a nice
        #    vectorized implementation.. We will just filter out the
        #    unnecessary slots later.
        split_vertices = np.repeat(vertices, 2, axis=0)
        split_vertices = np.concatenate(
            (
                split_vertices,
                np.full(
                    (split_vertices.shape[0], split_edges.shape[1], g.d),
                    np.nan,
                ),
            ),
            axis=1,
        )

        # Now we need to fill in the new vertices:
        # For each original tile vertex, we need to determine whether the tile
        # lies in the new null tile or the new alt tile.
        include_in_null_tile = vertex_dist >= -eps
        include_in_alt_tile = vertex_dist <= eps

        # Since we copied the entire tiles, we can "delete" vertices by
        # multiply by nan
        # note: ::2 traverses the range of new null hypo tiles
        #       1::2 traverses the range of new alt hypo tiles
        split_vertices[::2, : vertices.shape[1]] *= np.where(
            include_in_null_tile, 1, np.nan
        )[..., None]
        split_vertices[1::2, : vertices.shape[1]] *= np.where(
            include_in_alt_tile, 1, np.nan
        )[..., None]

        # The intersection vertices get added to both new tiles because
        # they lie on the boundary between the two tiles.
        split_vertices[::2, vertices.shape[1] :] = new_vs
        split_vertices[1::2, vertices.shape[1] :] = new_vs

        # Trim the new tile array:
        # We now are left with an array of tile vertices that has many more
        # vertex slots per tile than necessary with the unused slots filled
        # with nan.
        # To deal with this:
        # 1. We sort along the vertices axis. This has the effect of
        #    moving all the nan vertices to the end of the list.
        split_vertices = split_vertices[
            np.arange(split_vertices.shape[0])[:, None],
            np.argsort(np.sum(split_vertices, axis=-1), axis=-1),
        ]

        # 2. Identify the maximum number of vertices of any tile and trim the
        #    array so that is the new vertex dimension size
        nonfinite_corners = (~np.isfinite(split_vertices)).all(axis=(0, 2))
        # 3. If any corner is unused for all tiles, we should remove it.
        #    But, we can't trim smaller than the original vertices array.
        if nonfinite_corners[-1]:
            first_all_nan_corner = nonfinite_corners.argmax()
            split_vertices = split_vertices[:, :first_all_nan_corner]

        ########################################
        # Step 3. Identify bounding boxes.
        ########################################
        min_val = np.nanmin(split_vertices, axis=1)
        max_val = np.nanmax(split_vertices, axis=1)
        new_theta = (min_val + max_val) / 2
        new_radii = (max_val - min_val) / 2

        parent_id = np.repeat(g.df["id"], 2)
        g_split = grid.init_grid(new_theta, new_radii, g.worker_id, parents=parent_id)

        return g_split


def hypo(str_expr):
    """
    Define a hyperplane from a sympy expression.

    For example:
    >>> hypo("2*theta1 < 1")
    HyperPlane(n=array([ 0., -1.]), c=-0.5)

    >>> hypo("x - y >= 0")
    HyperPlane(n=array([ 0.70710678, -0.70710678]), c=0.0)

    Valid comparison operators are <, >, <=, >=.

    The left hand and right hand sides must be linear in theta.

    Aliases:
        - theta{i}: x{i}
        - x: x0
        - y: x1
        - z: x2

    Args:
        str_expr: The expression defining the hypothesis plane.

    Returns:
        The HyperPlane object corresponding to the sympy expression.
    """
    alias = dict(
        x="x0",
        y="x1",
        z="x2",
    )
    expr = sp.parsing.parse_expr(str_expr)
    if isinstance(expr, sp.StrictLessThan) or isinstance(expr, sp.LessThan):
        plane = expr.rhs - expr.lhs
    elif isinstance(expr, sp.StrictGreaterThan) or isinstance(expr, sp.GreaterThan):
        plane = expr.lhs - expr.rhs
    else:
        raise ValueError("Hypothesis expression must be an inequality.")

    symbols = plane.free_symbols
    coeffs = sp.Poly(plane, *symbols).coeffs()
    if len(coeffs) > len(symbols):
        c = -float(coeffs[-1])
        coeffs = coeffs[:-1]
    else:
        c = 0

    symbol_names = [alias.get(s.name, s.name).replace("theta", "x") for s in symbols]

    if any([s[0] != "x" for s in symbol_names]):
        raise ValueError(
            f"Hypothesis contains invalid symbols: {symbols}."
            " Valid symbols are x0..., theta0..., x, y, z."
        )
    try:
        symbol_idxs = [int(s[1:]) for s in symbol_names]
    except ValueError:
        raise ValueError(
            f"Hypothesis contains invalid symbols: {symbols}."
            " Valid symbols are x0..., theta0..., x, y, z."
        )
    coeff_dict = dict(zip(symbol_idxs, coeffs))
    max_idx = max(symbol_idxs)

    n = [float(coeff_dict.get(i, 0)) for i in range(max_idx + 1)]
    n_norm = np.linalg.norm(n)
    n /= n_norm
    c /= n_norm

    return HyperPlane(np.array(n), c)
