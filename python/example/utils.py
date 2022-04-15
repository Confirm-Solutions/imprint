import os
import pathlib
from datetime import timedelta
from logging import DEBUG as log_level
from logging import basicConfig, getLogger
from timeit import default_timer as timer

import numpy as np
from pykevlar.bound import TypeIErrorBound
from pykevlar.grid import Gridder, GridRange

basicConfig(
    level=log_level,
    format="%(asctime)s %(levelname)-8s %(module)-20s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = getLogger(__name__)

data_dir = "data"  # changeable


def to_array(v, size):
    if isinstance(v, float):
        v = [v]
    v = np.array(v * size) if len(v) == 1 else np.array(v)
    if v.shape[0] != size:
        raise ValueError(f"v (={v}) must be either dimension 1 or size (={size}).")
    return v


def make_grid(size, lower, upper):
    assert lower.shape[0] == upper.shape[0]

    # make initial 1d grid
    theta_grids = (
        Gridder.make_grid(size, lower[i], upper[i]) for i in range(len(lower))
    )
    # make corresponding radius
    radius = [Gridder.radius(size, lower[i], upper[i]) for i in range(len(lower))]
    return theta_grids, radius


def make_cartesian_grid_range(size, lower, upper, grid_sim_size):
    theta_grids, radius = make_grid(size, lower, upper)

    coords = np.meshgrid(*theta_grids)
    grid = np.concatenate([c.flatten().reshape(-1, 1) for c in coords], axis=1)
    gr = GridRange(grid.shape[1], grid.shape[0])

    gr.thetas()[...] = np.transpose(grid)

    radii = gr.radii()
    for i, row in enumerate(radii):
        row[...] = radius[i]

    gr.sim_sizes()[...] = grid_sim_size

    return gr


def save_ub(p_name, b_name, P, B):
    basepath = pathlib.Path(__file__).parent.resolve()
    datapath = os.path.join(basepath, data_dir)

    if not os.path.exists(datapath):
        os.makedirs(datapath)

    p_path = os.path.join(datapath, p_name)
    b_path = os.path.join(datapath, b_name)
    np.savetxt(p_path, P, fmt="%s", delimiter=",")
    np.savetxt(b_path, B, fmt="%s", delimiter=",")


def create_ub_plot_inputs(model, acc_o, gr, delta):
    assert model.n_models() == 1
    ub = TypeIErrorBound()
    kbs = model.make_kevlar_bound_state(gr)

    start = timer()
    ub.create(kbs, acc_o, gr, delta)
    end = timer()
    logger.info("Kevlar bound time: {}".format(timedelta(seconds=end - start)))

    P = []
    B = []
    pos = 0
    for i in range(gr.n_gridpts()):
        for j in range(gr.n_tiles(i)):
            P.append(gr.thetas()[:, i])
            B.append(
                [
                    ub.delta_0()[0, pos],
                    ub.delta_0_u()[0, pos],
                    ub.delta_1()[0, pos],
                    ub.delta_1_u()[0, pos],
                    ub.delta_2_u()[0, pos],
                    ub.get()[0, pos],
                ]
            )
            pos += 1
    return np.array(P).T, np.array(B)
