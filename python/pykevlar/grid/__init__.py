import numpy as np

# TODO: note that AdaGridInternal really should not be exposed.
from pykevlar.core.grid import *
from pykevlar.core.grid import Gridder, GridRange
from pykevlar.grid.adagrid import AdaGrid


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
