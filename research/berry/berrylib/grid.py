import warnings
from dataclasses import dataclass
from itertools import product
from typing import List

import numpy as np


@dataclass
class HyperPlane:
    n: np.ndarray
    c: float


@dataclass
class Grid:
    thetas: np.ndarray
    radii: np.ndarray
    vertices: np.ndarray
    is_regular: np.ndarray
    null_truth: np.ndarray
    grid_pt_idx: np.ndarray


def build_grid(thetas: np.ndarray, radii: np.ndarray, null_hypos: List[HyperPlane]):
    n_params = thetas.shape[1]

    edges = get_edges(thetas, radii)

    unit_vs = hypercube_vertices(n_params)
    tile_vertices = thetas[:, None, :] + (unit_vs[None, :, :] * radii[:, None, :])

    grid_pt_idx = np.arange(thetas.shape[0])
    is_regular = np.ones(thetas.shape[0], dtype=bool)
    null_truth = np.full((tile_vertices.shape[0], len(null_hypos)), -1)
    eps = 1e-15

    history = []
    for iH, H in enumerate(null_hypos):
        max_v_count = tile_vertices.shape[1]

        dist = tile_vertices.dot(H.n) - H.c

        # -1 means split, 0 means alt true, 1 means null true
        is_null = ((dist >= 0) | np.isnan(dist)).all(axis=1)
        null_truth[is_null, iH] = 1
        null_truth[~is_null, iH] = 0

        # Identify the tiles to be split
        to_split = ~(
            ((dist >= -eps) | np.isnan(dist)).all(axis=1)
            | ((dist <= eps) | np.isnan(dist)).all(axis=1)
        )

        # Irregular tiles are not split, just copied. This is just a simplification.
        split_or_copy_idxs = np.where(to_split)[0]
        split_idxs = np.where(to_split & is_regular)[0]
        print(split_idxs.shape[0])

        # # Intersect every tile edge with the hyperplane to find the new vertices.
        split_edges = edges[grid_pt_idx[split_idxs]]
        split_vs = split_edges[..., :2]
        split_dir = split_edges[..., 2:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha = (H.c - split_vs.dot(H.n)) / (split_dir.dot(H.n))
            new_vs = split_vs[:, :max_v_count] + alpha[:, :, None] * split_dir
            # The non-finite values may be either nan or inf. Here, we homogenize them
            # to all be nan. just simplifies later checks.
            new_vs = np.where(
                (np.isfinite(new_vs)) & ((alpha > eps) & (alpha < 1 - eps))[..., None],
                new_vs,
                np.nan,
            )

        # Create the new tiles with a slot for an intersection from each edge. This
        # is more slots than necessary. We fill the unused space with nan and
        # filter those slots out later.
        new_tiles = np.repeat(tile_vertices, to_split + 1, axis=0)
        new_tiles = np.concatenate(
            (
                new_tiles,
                np.full((new_tiles.shape[0], edges.shape[1], n_params), np.nan),
            ),
            axis=1,
        )

        is_regular = np.repeat(is_regular, to_split + 1)
        new_split_or_copy_idxs = split_or_copy_idxs + np.arange(
            split_or_copy_idxs.shape[0]
        )
        new_split_idxs = new_split_or_copy_idxs[is_regular[new_split_or_copy_idxs]]
        is_regular[new_split_or_copy_idxs] = False
        is_regular[new_split_or_copy_idxs + 1] = False
        np.testing.assert_allclose(
            new_tiles[new_split_idxs, :max_v_count], tile_vertices[split_idxs]
        )

        include_in_null_tile = dist[split_idxs] >= -eps
        include_in_alt_tile = dist[split_idxs] <= eps
        new_tiles[new_split_idxs, :max_v_count] *= np.where(
            include_in_null_tile, 1, np.nan
        )[..., None]
        new_tiles[new_split_idxs + 1, :max_v_count] *= np.where(
            include_in_alt_tile, 1, np.nan
        )[..., None]
        new_tiles[new_split_idxs, max_v_count:] = new_vs
        new_tiles[new_split_idxs + 1, max_v_count:] = new_vs
        new_tiles.sort(axis=1)

        # Trim the new tile array
        finite_corners = (~np.isfinite(new_tiles)).all(axis=(0, 2))
        if finite_corners[-1]:
            first_all_nan_corner = finite_corners.argmax()
            new_tiles = new_tiles[:, :first_all_nan_corner]

        parents = np.repeat(np.arange(tile_vertices.shape[0]), to_split + 1)
        history.append(
            dict(
                parents=parents,
                vertices=tile_vertices.copy(),
                split_vs=split_vs,
                split_dir=split_dir,
                split_idxs=split_idxs,
                alpha=alpha,
            )
        )

        tile_vertices = new_tiles
        grid_pt_idx = np.repeat(grid_pt_idx, to_split + 1)
        null_truth = np.repeat(null_truth, to_split + 1, axis=0)
        null_truth[new_split_or_copy_idxs, iH] = 1
        null_truth[new_split_or_copy_idxs + 1, iH] = 0

    return Grid(thetas, radii, tile_vertices, is_regular, null_truth, grid_pt_idx)


def prune(g):
    if g.null_truth.shape[1] == 0:
        return g
    all_alt = (g.null_truth == 0).all(axis=1)
    grid_pt_idx = g.grid_pt_idx[~all_alt]
    included_grid_pts, grid_pt_inverse = np.unique(grid_pt_idx, return_inverse=True)
    return Grid(
        g.thetas[included_grid_pts],
        g.radii[included_grid_pts],
        g.vertices[~all_alt],
        g.is_regular[~all_alt],
        g.null_truth[~all_alt],
        grid_pt_inverse,
    )


# https://stackoverflow.com/a/52229385/
def hypercube_vertices(d):
    """
    The corners of a hypercube of dimension d.

    print(vertices(1))
    >>> [(1,), (-1,)]

    print(vertices(2))
    >>> [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    print(vertices(3))
    >>> [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
    ]
    """
    return np.array(list(product((1, -1), repeat=d)))


def get_edges(thetas, radii):
    n_params = thetas.shape[1]
    unit_vs = hypercube_vertices(n_params)
    n_vs = unit_vs.shape[0]
    unit_edges = []
    for i in range(n_vs):
        for j in range(n_params):
            if unit_vs[i, j] > 0:
                continue
            unit_edges.append(np.concatenate((unit_vs[i], np.identity(n_params)[j])))

    # edges[:, :, :n_params] are the vertices at the origin of the edges
    # edges[:, :, n_params:] are the edge vectors pointing from the start to
    # the end of the edge
    # in total, the edges array has shape:
    # (n_grid_pts, number of hypercube vertices, 2*n_params)
    edges = np.tile(np.array(unit_edges)[None, :, :], (thetas.shape[0], 1, 1))
    edges[:, :, :n_params] *= radii[:, None, :]
    edges[:, :, n_params:] *= 2 * radii[:, None, :]
    edges[:, :, :n_params] += thetas[:, None, :]
    return edges
