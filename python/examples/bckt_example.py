import pykevlar.core as core
import pykevlar.driver as driver
import numpy as np
import os
from utils import (
    save_ub,
    create_ub_plot_inputs,
)

# ========== Toggleable ===============
n_arms = 2
sim_size = 10000
n_thetas_1d = 64
n_threads = os.cpu_count()
# ========== End Toggleable ===============

ph2_size = 50
n_samples = 250
seed = 69
thresh = 2.1
lower = -0.5
upper = 0.5
delta = 0.025

# set numpy random seed
np.random.seed(seed)

# define null hypos
null_hypos = []
for i in range(1, n_arms):
    n = np.zeros(n_arms)
    n[0] = 1
    n[i] = -1
    null_hypos.append(core.HyperPlane(n, 0))

# Create current batch of grid points.
# At the process-level, we only need to know theta, radii.
theta_1d = core.Gridder.make_grid(n_thetas_1d, lower, upper)
grid = np.stack(np.meshgrid(*(theta_1d for _ in range(n_arms))), axis=-1) \
        .reshape(-1, n_arms)
gr = core.GridRange(n_arms, grid.shape[0])
thetas = gr.thetas()
thetas[...] = np.transpose(grid)
radii = gr.radii()
radii[...] = core.Gridder.radius(n_thetas_1d, lower, upper)
sim_sizes = gr.sim_sizes()
sim_sizes[...] = sim_size

gr.create_tiles(null_hypos)
gr.prune()

# create BCKT
bckt = core.BinomialControlkTreatment(n_arms, ph2_size, n_samples, [thresh])

# run a mock-call of fit_process
is_o = driver.fit_process(bckt, gr, sim_size, seed, n_threads)

# create upper bound plot inputs and save info
P, B = create_ub_plot_inputs(bckt, is_o, gr, delta)
save_ub(
    'P-bckt-6eeba17dfc476eea1cc7f562d367e5026112d4fb.csv',
    'B-bckt-6eeba17dfc476eea1cc7f562d367e5026112d4fb.csv',
    P,
    B,
)
