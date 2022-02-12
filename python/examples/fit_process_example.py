import pykevlar.core as core
import pykevlar.driver as driver
import numpy as np
import os
import timeit

n_arms = 3
ph2_size = 50
n_samples = 250
sim_size = 100000
n_thetas_1d = 64
seed = 69
thresh = 1.96
n_threads = os.cpu_count()
lower = -0.5
upper = 0.5

# set numpy random seed
np.random.seed(seed)

# define null hypos
def null_hypo(i, p):
    return p[i] <= p[0]

# Create current batch of grid points.
# At the process-level, we only need to know theta, radii.
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

# create BCKT
bckt = core.BinomialControlkTreatment(n_arms, ph2_size, n_samples, thresh)
bckt.set_grid_range(gr, null_hypo)

# run a mock-call of fit_process
is_o = driver.fit_process(bckt, sim_size, seed, n_threads)
print((is_o.type_I_sum() / is_o.n_accum())[0,:20])
