from types import MethodType

import numpy as np
from pyimprint.core.grid import *
from pyimprint.core.grid import Gridder
from pyimprint.core.grid import GridRange
from pyimprint.grid.adagrid import AdaGrid

# TODO: note that AdaGridInternal really should not be exposed.


def n_tiles_per_pt(gr):
    cum_n_tiles = np.array(gr.cum_n_tiles())
    return cum_n_tiles[1:] - cum_n_tiles[:-1]


def theta_tiles(gr):
    return np.repeat(gr.thetas().T, n_tiles_per_pt(gr), axis=0)


def radii_tiles(gr):
    return np.repeat(gr.radii().T, n_tiles_per_pt(gr), axis=0)


def sim_sizes_tiles(gr):
    return np.repeat(gr.sim_sizes(), n_tiles_per_pt(gr), axis=0)


def is_null_per_arm(gr):
    tiles = theta_tiles(gr)
    n_arms = tiles.shape[-1]
    return np.array(
        [[gr.check_null(i, j) for j in range(n_arms)] for i in range(tiles.shape[0])]
    )


def collect_corners(gr):
    # gr.corners expects a 2D array with shape: (n_tiles * 2^(d+1), d)
    # unfilled indices will left as nan in order to be easily filtered out
    # later on.
    # We pass 2^(d+1) corner slots for each tile since that is guaranteed to be
    # greater than the true number of corners. Since we only split a tile once,
    # the true maximum should actually be (2^d) + d - 1.
    corners = np.full((gr.n_tiles() * 2 ** (gr.n_params() + 1), gr.n_params()), np.nan)

    # gr.corners(...) fills the corners array in place.
    gr.corners(corners)

    # After this reshape, the corners array will be: (n_tiles, 2^(d+1), d)
    # Then, we will remove any corner indices that are entirely nan. After this
    # loop the second dimension will be reduced in length from 2^(d+1) to the
    # maximum number of corners for any tile.
    corners = corners.reshape((gr.n_tiles(), -1, gr.n_params()))
    for i in range(2 ** (gr.n_params() + 1)):
        if np.all(np.isnan(corners[:, i, :])):
            corners = corners[:, :i]
            break
    return corners


def make_cartesian_grid_range(size, lower, upper, grid_sim_size):
    assert lower.shape[0] == upper.shape[0]

    # make initial 1d grid
    theta_grids = (
        Gridder.make_grid(size, lower[i], upper[i]) for i in range(len(lower))
    )
    # make corresponding radius
    radius = [Gridder.radius(size, lower[i], upper[i]) for i in range(len(lower))]

    coords = np.meshgrid(*theta_grids)
    grid = np.concatenate([c.flatten().reshape(-1, 1) for c in coords], axis=1)
    gr = GridRange(grid.shape[1], grid.shape[0])

    gr.thetas()[...] = np.transpose(grid)

    radii = gr.radii()
    for i, row in enumerate(radii):
        row[...] = radius[i]

    gr.sim_sizes()[...] = grid_sim_size

    return gr
