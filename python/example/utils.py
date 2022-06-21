import logging
import os
import pathlib
from datetime import timedelta
from timeit import default_timer as timer

import numpy as np
from pyimprint.bound import TypeIErrorBound

log_level = logging.DEBUG
logging.basicConfig(
    level=log_level,
    format="%(asctime)s %(levelname)-8s %(module)-20s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# Disable matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)

data_dir = "data"  # changeable


def to_array(v, size):
    if isinstance(v, float):
        v = [v]
    v = np.array(v * size) if len(v) == 1 else np.array(v)
    if v.shape[0] != size:
        raise ValueError(f"v (={v}) must be either dimension 1 or size (={size}).")
    return v


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
    kbs = model.make_imprint_bound_state(gr)

    start = timer()
    ub.create(kbs, acc_o, gr, delta)
    end = timer()
    logger.info("Imprint bound time: {}".format(timedelta(seconds=end - start)))

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
