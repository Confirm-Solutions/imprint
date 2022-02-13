import pykevlar.core as core
import pykevlar.driver as driver
from pykevlar.batcher import SimpleBatch
import numpy as np
import os
import timeit

# ========== Toggleable ===============
n_arms = 3      # prioritize 3 first, then do 4
sim_size = 100000
n_thetas_1d = 64
n_threads = os.cpu_count()
max_batch_size = 64000
# ========== End Toggleable ===============

ph2_size = 50
n_samples = 250
seed = 69
thresh = 1.96
lower = -0.5
upper = 0.5

# set numpy random seed
np.random.seed(seed)

# define null hypos
def null_hypo(i, p):
    return p[i] <= p[0]

# Create full grid.
# At the driver-level, we need to know theta, radii, sim_sizes.
theta_1d = core.Gridder.make_grid(n_thetas_1d, lower, upper)
grid = np.stack(np.meshgrid(*(theta_1d for _ in range(n_arms))), axis=-1) \
        .reshape(-1, n_arms)
grid_null = np.array([
    p for p in grid if null_hypo(1, p) or null_hypo(2, p)
])
gr = core.GridRange(n_arms, grid_null.shape[0])
thetas = gr.get_thetas()
thetas[...] = np.transpose(grid_null)
radii = gr.get_radii()
radii[...] = core.Gridder.radius(n_thetas_1d, lower, upper)
sim_sizes = gr.get_sim_sizes()
sim_sizes[...] = sim_size

# create batcher
batcher = SimpleBatch(gr, max_batch_size)

# create BCKT
bckt = core.BinomialControlkTreatment(n_arms, ph2_size, n_samples, thresh)

# run a mock-call of fit_driver
# Currently, it will yield each batched result.
# TODO: once this doesn't yield anymore, modify this part.
for is_o in driver.fit_driver(batcher, null_hypo, bckt, seed, n_threads):
    print(is_o.type_I_sum() / is_o.n_accum())
