import warnings
from dataclasses import dataclass
from itertools import product
from typing import List

import numpy as np


@dataclass
class HyperPlane:
    """A plane defined by:
    x \cdot n + c = 0
    """

    n: np.ndarray
    c: float


@dataclass
class Grid:
    """
    The first two arrays define the grid points/cells:
    - thetas: the center of each hyperrectangle.
    - radii: the half-width of each hyperrectangle in each dimension.
        (NOTE: we could rename this since it's sort of a lie.)

    The next four arrays define the tiles:
    - vertices contains the vertices of each tiles. After splitting, tiles
      may have differing numbers of vertices. The vertices array will be
      shaped: (n_tiles, max_n_vertices, n_params). For tiles that have fewer
      than max_n_vertices, the unused entries will be filled with nans.
    - grid_pt_idx is an array with an entry for each tile that contains to
      index of the original grid point from which that tile was created
    - is_regular indicates whether each tile has ever been split. Tiles that
      have been split are considered "irregular" and tiles that have never been
      split are considered "regular".
    - null_truth indicates the truth of each null hypothesis for each tile.
    """

    thetas: np.ndarray
    radii: np.ndarray
    vertices: np.ndarray
    is_regular: np.ndarray
    null_truth: np.ndarray
    grid_pt_idx: np.ndarray

    @property
    def n_tiles(self):
        return self.vertices.shape[0]

    @property
    def n_grid_pts(self):
        return self.thetas.shape[0]


def build_grid(
    thetas: np.ndarray, radii: np.ndarray, null_hypos: List[HyperPlane], debug=False
):
    """
    Construct a Imprint grid from a set of grid point centers, radii and null
    hypothesis.
    1. Initially, we construct simple hyperrectangle cells.
    2. Then, we split cells that are intersected by the null hypothesis boundaries.

    Note that we do not split cells twice. This is a simplification that makes
    the software much simpler and probably doesn't cost us much in terms of
    bound tightness because very few cells are intersected by multiple
    hyperplanes.

    Parameters
    ----------
    thetas
        The centers of the hyperrectangle grid.
    radii
        The half-width of each hyperrectangle in each dimension.
    null_hypos
        A list of hyperplanes defining the boundary of the null hypothesis. The
        normal vector of these hyperplanes point into the null domain.


    Returns
    -------
        a Grid object
    """
    n_grid_pts, n_params = thetas.shape

    # For splitting cells, we will need to know the nD edges of each cell and
    # the vertices of each tile.
    edges = get_edges(thetas, radii)
    unit_vs = hypercube_vertices(n_params)
    tile_vs = thetas[:, None, :] + (unit_vs[None, :, :] * radii[:, None, :])

    # Keep track of the various tile properties. See the Grid class docstring
    # for definitions.
    grid_pt_idx = np.arange(n_grid_pts)
    is_regular = np.ones(n_grid_pts, dtype=bool)
    null_truth = np.full((n_grid_pts, len(null_hypos)), -1)
    eps = 1e-15

    history = []
    for iH, H in enumerate(null_hypos):
        max_v_count = tile_vs.shape[1]

        # Measure the distance of each vertex from the null hypo boundary
        # 0 means alt true, 1 means null true
        # it's important to allow nan dist because some tiles may not have
        # every vertex slot filled. unused vertex slots will contain nans.
        dist = tile_vs.dot(H.n) - H.c
        is_null = ((dist >= 0) | np.isnan(dist)).all(axis=1)
        null_truth[is_null, iH] = 1
        null_truth[~is_null, iH] = 0

        # Identify the tiles to be split. Give some floating point slack around
        # zero so we don't suffer from imprecision.
        to_split = ~(
            ((dist >= -eps) | np.isnan(dist)).all(axis=1)
            | ((dist <= eps) | np.isnan(dist)).all(axis=1)
        )

        # Track which tile indices will be split or copied.
        # Tiles that have already been split ("irregular tiles") are not split,
        # just copied. This is just a simplification that makes the software
        # much simpler
        split_or_copy_idxs = np.where(to_split)[0]
        split_idxs = np.where(to_split & is_regular)[0]

        # Intersect every tile edge with the hyperplane to find the new vertices.
        split_edges = edges[grid_pt_idx[split_idxs]]
        # The first n_params columns of split_edges are the vertices from which
        # the edge originates and the second n_params are the edge vector.
        split_vs = split_edges[..., :n_params]
        split_dir = split_edges[..., n_params:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Intersect each edge with the plane.
            alpha = (H.c - split_vs.dot(H.n)) / (split_dir.dot(H.n))
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

        # Create the array for the new vertices. We expand the original tile_vs
        # array in both dimensions:
        # 1. We create a new row for each tile that is being split using np.repeat.
        # 2. We create a new column for each potential additional vertex from
        #    the intersection operation above using np.concatenate. This is far
        #    more new vertices than necessary, but facilitates a nice vectorized
        #    implementation.. We will just filter out the unnecessary slots later.
        # (note: to_split + 1 will be 1 for each unsplit tile and 2 for each
        # split tile, so this np.repeat will duplicated rows that are
        # being split)
        new_tile_vs = np.repeat(tile_vs, to_split + 1, axis=0)
        new_tile_vs = np.concatenate(
            (
                new_tile_vs,
                np.full((new_tile_vs.shape[0], edges.shape[1], n_params), np.nan),
            ),
            axis=1,
        )

        # For each split tile, we need the indices of the tiles *after* the
        # creation of the new array. This will be the existing index plus the
        # count of lower-index split tiles.
        new_split_or_copy_idxs = split_or_copy_idxs + np.arange(
            split_or_copy_idxs.shape[0]
        )
        is_regular = np.repeat(is_regular, to_split + 1)
        new_split_idxs = new_split_or_copy_idxs[is_regular[new_split_or_copy_idxs]]
        # Update the is_regular array:
        # - split tiles are marked irregular.
        is_regular[new_split_or_copy_idxs] = False
        is_regular[new_split_or_copy_idxs + 1] = False
        np.testing.assert_allclose(
            new_tile_vs[new_split_idxs, :max_v_count], tile_vs[split_idxs]
        )

        # For each original tile vertex, we need to determine whether the tile
        # lies in the new null tile or the new alt tile.
        include_in_null_tile = dist[split_idxs] >= -eps
        include_in_alt_tile = dist[split_idxs] <= eps

        # Since we copied the entire tiles, we can "delete" vertices by multiply by nan
        # note: new_split_idxs     marks the index of the new null tile
        #       new_split_idxs + 1 marks the index of the new alt  tile
        new_tile_vs[new_split_idxs, :max_v_count] *= np.where(
            include_in_null_tile, 1, np.nan
        )[..., None]
        new_tile_vs[new_split_idxs + 1, :max_v_count] *= np.where(
            include_in_alt_tile, 1, np.nan
        )[..., None]
        # The intersection vertices get added to both new tiles.
        new_tile_vs[new_split_idxs, max_v_count:] = new_vs
        new_tile_vs[new_split_idxs + 1, max_v_count:] = new_vs

        # Trim the new tile array:
        # We now are left with an array of tile vertices that has many more
        # vertex slots per tile than necessary with the unused slots filled
        # with nan.
        # To deal with this:
        # 1. We sort along the vertices axis. This has the effect of
        #    moving all the nan vertices to the end of the list.
        new_tile_vs.sort(axis=1)
        # 2. Identify the maximum number of vertices of any tile and trim the
        #    array so that is the new vertex dimension size
        finite_corners = (~np.isfinite(new_tile_vs)).all(axis=(0, 2))
        if finite_corners[-1]:
            first_all_nan_corner = finite_corners.argmax()
            new_tile_vs = new_tile_vs[:, :first_all_nan_corner]

        # For debugging purposes, it can be helpful to track the parent tile
        # index of each new tile.
        if debug:
            parents = np.repeat(np.arange(tile_vs.shape[0]), to_split + 1)

        # Hurray, we made it! Replace the tile array!
        tile_vs = new_tile_vs

        # Update the remaining tile characteristics.
        # - the two sides of a split tile have their null hypo truth indicators updated.
        null_truth = np.repeat(null_truth, to_split + 1, axis=0)
        null_truth[new_split_or_copy_idxs, iH] = 1
        null_truth[new_split_or_copy_idxs + 1, iH] = 0
        # - duplicate the reference to the original grid pt for each split tile.
        grid_pt_idx = np.repeat(grid_pt_idx, to_split + 1)

        # Data on the intermediate state of the splitting can be helpful for
        # debugging.
        if debug:
            history.append(
                dict(
                    parents=parents,
                    split_vs=split_vs,
                    split_dir=split_dir,
                    split_idxs=split_idxs,
                    alpha=alpha,
                    grid=Grid(
                        thetas, radii, tile_vs, is_regular, null_truth, grid_pt_idx
                    ),
                )
            )

    out = Grid(thetas, radii, tile_vs, is_regular, null_truth, grid_pt_idx)
    if debug:
        return out, history
    else:
        return out


def prune(g):
    """Remove tiles that are entirely within the alternative hypothesis space.

    Parameters
    ----------
    g
        the Grid object

    Returns
    -------
        the pruned Grid object.
    """
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
    """
    Construct an array indicating the edges of each hyperrectangle.
    - edges[:, :, :n_params] are the vertices at the origin of the edges
    - edges[:, :, n_params:] are the edge vectors pointing from the start to
        the end of the edge

    In total, the edges array has shape:
    (n_grid_pts, number of hypercube vertices, 2*n_params)
    """

    n_params = thetas.shape[1]
    unit_vs = hypercube_vertices(n_params)
    n_vs = unit_vs.shape[0]
    unit_edges = []
    for i in range(n_vs):
        for j in range(n_params):
            if unit_vs[i, j] > 0:
                continue
            unit_edges.append(np.concatenate((unit_vs[i], np.identity(n_params)[j])))

    edges = np.tile(np.array(unit_edges)[None, :, :], (thetas.shape[0], 1, 1))
    edges[:, :, :n_params] *= radii[:, None, :]
    edges[:, :, n_params:] *= 2 * radii[:, None, :]
    edges[:, :, :n_params] += thetas[:, None, :]
    return edges
