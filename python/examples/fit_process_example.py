from pykevlar.grid import (
    GridRange,
    HyperPlane,
    Gridder,
)
from pykevlar.model.binomial import SimpleSelection
from pykevlar.driver import accumulate_process
import numpy as np
import os
from datetime import timedelta
from timeit import default_timer as timer

import numpy as np
import pykevlar.core as core
import pykevlar.driver as driver

# ========== Toggleable ===============
n_arms = 3  # prioritize 3 first, then do 4
sim_size = 100000
n_thetas_1d = 64
n_threads = os.cpu_count()
# ========== End Toggleable ===============

ph2_size = 50
n_samples = 250
seed = 69
thresh = 2.1
lower = -0.5
upper = 0.5

# set numpy random seed
np.random.seed(seed)

# define null hypos
null_hypos = []
for i in range(1, n_arms):
    n = np.zeros(n_arms)
    n[0] = 1
    n[i] = -1
    null_hypos.append(HyperPlane(n, 0))

# Create current batch of grid points.
# At the process-level, we only need to know theta, radii.
theta_1d = Gridder.make_grid(n_thetas_1d, lower, upper)
grid = np.stack(np.meshgrid(*(theta_1d for _ in range(n_arms))), axis=-1) \
        .reshape(-1, n_arms)
gr = GridRange(n_arms, grid.shape[0])
thetas = gr.thetas()
thetas[...] = np.transpose(grid)
radii = gr.radii()
radii[...] = Gridder.radius(n_thetas_1d, lower, upper)

gr.create_tiles(null_hypos)

start = timer()
gr.prune()
end = timer()

print("Prune time: {t}".format(t=timedelta(seconds=end-start)))
print("Gridpts: {n}".format(n=gr.n_gridpts()))
print("Tiles: {n}".format(n=gr.n_tiles()))

# create model
model = SimpleSelection(n_arms, n_samples, ph2_size, [thresh])

# run a mock-call of accumulate_process
start = timer()
out = accumulate_process(model, gr, sim_size, seed, n_threads)
end = timer()

print("Fit time: {t}".format(t=timedelta(seconds=end-start)))
print((out.typeI_sum() / sim_size))
