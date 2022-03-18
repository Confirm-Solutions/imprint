import os
import pathlib
import numpy as np
import pykevlar.core as core

data_dir = 'data'   # changeable

basepath = pathlib.Path(__file__).parent.resolve()
datapath = os.path.join(basepath, data_dir)
if not os.path.exists(datapath):
    os.makedirs(datapath)


def save_ub(p_name, b_name, P, B):
    p_path = os.path.join(datapath, p_name)
    b_path = os.path.join(datapath, b_name)
    np.savetxt(p_path, P, fmt='%s', delimiter=',')
    np.savetxt(b_path, B, fmt='%s', delimiter=',')


def create_ub_plot_inputs(model, is_o, gr, delta):
    ub = core.UpperBound()
    ub.create(model, is_o, gr, delta)
    P = []
    B = []
    pos = 0
    for i in range(gr.n_gridpts()):
        for j in range(gr.n_tiles(i)):
            P.append(gr.thetas()[:, i])
            B.append([
                ub.delta_0()[0, pos],
                ub.delta_0_u()[0, pos],
                ub.delta_1()[0, pos],
                ub.delta_1_u()[0, pos],
                ub.delta_2_u()[0, pos],
                ub.get()[0, pos],
            ])
            pos += 1
    return np.array(P).T, np.array(B)
