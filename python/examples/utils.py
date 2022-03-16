import os
import pathlib
import numpy as np

data_dir = 'data'   # changeable

basepath = pathlib.Path(__file__).parent.resolve()
datapath = os.path.join(basepath, data_dir)
if not os.path.exists(datapath):
    os.makedirs(datapath)


def save_ub(p_name, b_name, P, B):
    p_path = os.path.join(datapath, p_name)
    b_path = os.path.join(datapath, b_name)
    np.savetxt(p_path, P, fmt='%s')
    np.savetxt(b_path, B, fmt='%s')
