import pykevlar.core as core
import pykevlar.driver as driver
import numpy as np
import os
from utils import (
    save_ub,
    create_ub_plot_inputs,
)

# ========== Toggleable ===============
sim_size = 10000
n_lmda_1d = 64
n_hzrd_1d = 64
n_threads = os.cpu_count()
# ========== End Toggleable ===============

n_arms = 2
n_samples = 250
seed = 69
thresh = 1.96
log_lmda_lower = -0.1/4
log_lmda_upper = 1./4
log_hzrd_lower = log_lmda_lower-log_lmda_upper
log_hzrd_upper = 0.0
censor_time = 2.0
delta = 0.025

# set numpy random seed
np.random.seed(seed)

# define null hypos
null_hypos = [core.HyperPlane(np.array([0, -1]), 0)]

# Create full grid.
# At the driver-level, we need to know theta, radii, sim_sizes.
lmda_1d = core.Gridder.make_grid(n_lmda_1d, log_lmda_lower, log_lmda_upper)
hzrd_1d = core.Gridder.make_grid(n_hzrd_1d, log_hzrd_lower, log_hzrd_upper)
grid = np.stack(np.meshgrid(lmda_1d, hzrd_1d), axis=-1) \
        .reshape(-1, n_arms)
gr = core.GridRange(n_arms, grid.shape[0])
thetas = gr.thetas()
thetas[...] = np.transpose(grid)
radii = gr.radii()
radii[0, :] = core.Gridder.radius(n_lmda_1d, log_lmda_lower, log_lmda_upper)
radii[1, :] = core.Gridder.radius(n_hzrd_1d, log_hzrd_lower, log_hzrd_upper)
sim_sizes = gr.sim_sizes()
sim_sizes[...] = sim_size

gr.create_tiles(null_hypos)
gr.prune()

# create ECKT
eckt = core.ExpControlkTreatment(n_samples, censor_time, [thresh])

# fit on current grid range and output csv files for upper bound
is_o = driver.fit_process(eckt, gr, sim_size, seed, n_threads)

# create upper bound plot inputs and save info
P, B = create_ub_plot_inputs(eckt, is_o, gr, delta)
save_ub(
    'P-eckt-6eeba17dfc476eea1cc7f562d367e5026112d4fb.csv',
    'B-eckt-6eeba17dfc476eea1cc7f562d367e5026112d4fb.csv',
    P,
    B,
)
