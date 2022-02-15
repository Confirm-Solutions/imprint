import pykevlar.core as core
import pykevlar.driver as driver
from pykevlar.batcher import SimpleBatch
import numpy as np
import os
import timeit

# ========== Toggleable ===============
sim_size = 100000
n_lmda_1d = 64
n_hzrd_1d = 64
n_threads = os.cpu_count()
max_batch_size = -1
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

# set numpy random seed
np.random.seed(seed)

# define null hypos
def null_hypo(p):
    hzrd_rate = p[1]
    return hzrd_rate <= 1

# Create full grid.
# At the driver-level, we need to know theta, radii, sim_sizes.
lmda_1d = core.Gridder.make_grid(n_lmda_1d, log_lmda_lower, log_lmda_upper)
hzrd_1d = core.Gridder.make_grid(n_hzrd_1d, log_hzrd_lower, log_hzrd_upper)
grid = np.stack(np.meshgrid(lmda_1d, hzrd_1d), axis=-1) \
        .reshape(-1, n_arms)
grid_null = np.array([
    p for p in grid if p[1] <= 0
])
gr = core.GridRange(n_arms, grid_null.shape[0])
thetas = gr.get_thetas()
thetas[...] = np.transpose(grid_null)
radii = gr.get_radii()
radii[0,:] = core.Gridder.radius(n_lmda_1d, log_lmda_lower, log_lmda_upper)
radii[1,:] = core.Gridder.radius(n_hzrd_1d, log_hzrd_lower, log_hzrd_upper)
sim_sizes = gr.get_sim_sizes()
sim_sizes[...] = sim_size

# create batcher
batcher = SimpleBatch(gr, max_batch_size)

# create ECKT
eckt = core.ExpControlkTreatment(n_samples, censor_time, thresh)

# run a mock-call of fit_driver
# Currently, it will yield each batched result.
# TODO: once this doesn't yield anymore, modify this part.
for is_o in driver.fit_driver(batcher, null_hypo, eckt, seed, n_threads):
    print(is_o.type_I_sum() / sim_size)
