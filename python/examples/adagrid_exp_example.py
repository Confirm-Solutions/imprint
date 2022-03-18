from pykevlar.core import (
    GridRange,
    HyperPlane,
    ExpControlkTreatment,
)
from pykevlar.driver import (
    fit_process
)
from pykevlar.grid import AdaGrid
from pykevlar.batcher import SimpleBatch
from scipy.stats import norm
import numpy as np
import os

from logging import basicConfig, getLogger
from logging import DEBUG as log_level
logger = getLogger(__name__)

# ========== Toggleable ===============
max_iter = 15       # max iterations into adagrid
N_max = int(1E5)    # max simulation size
n_threads = os.cpu_count()
max_batch_size = 100000

logger.info("max_iter: %d, N_max: %d, "
            "n_threads: %d, max_batch_size: %d" %
            (max_iter,
             N_max, n_threads, max_batch_size))
# ========== End Toggleable ===============

n_arms = 2
init_sim_size = int(1E3) # initial simulation size
                    # (for simplicity, fixed for all points)
init_size = 8       # initial number of points along each direction
lower = np.array([-0.1/4, -1])
upper = np.array([1/4, 0])
alpha = 0.025
delta = 0.025
seed = 21324
ph2_size = 50
n_samples = 250
finalize_thr = alpha * 1.1

# TODO: temporary values to feed.
alpha_minus = alpha - 2*np.sqrt(alpha*(1-alpha)/init_sim_size)
thr = norm.isf(alpha)
thr_minus = norm.isf(alpha_minus)

# define null-hypo
null_hypos = [HyperPlane(np.array([0,-1]), 0)]

# make initial 1d grid
rnge = upper-lower
radius = rnge / (2*init_size)
theta_grids = (
    np.arange(lower[i]+radius[i], upper[i], step=2*radius[i])
    for i in range(len(lower))
)

# make full grid
coords = np.meshgrid(*theta_grids)
grid = np.concatenate(
    [c.flatten().reshape(-1,1) for c in coords],
    axis=1)

# create initial grid range
n_init_gridpts = grid.shape[0]
gr = GridRange(n_arms, n_init_gridpts)
thetas = gr.thetas()
thetas[...] = np.transpose(grid)
radii = gr.radii()
for i, row in enumerate(radii):
    row[...] = radius[i]
sim_sizes = gr.sim_sizes()
sim_sizes[...] = init_sim_size

# create model
model = ExpControlkTreatment(n_samples, 2.0, [])

# create batcher
batcher = SimpleBatch(max_size=max_batch_size)

adagrid = AdaGrid()
gr_new = adagrid.fit(
    batcher=batcher,
    model=model,
    null_hypos=null_hypos,
    init_grid=gr,
    alpha=alpha,
    delta=delta,
    seed=seed,
    max_iter=max_iter,
    N_max=N_max,
    alpha_minus=alpha_minus,
    thr=thr,
    thr_minus=thr_minus,
    finalize_thr=finalize_thr,
    rand_iter=False,
    debug=True,
)

import matplotlib.pyplot as plt

finals = None
curr = None
do_plot = True

i = 0
while 1:
    try:
        curr, finals = next(gr_new)
    except StopIteration:
        break

    if do_plot:
        thetas = curr.thetas_const()

        plt.scatter(thetas[0,:], thetas[1,:],
                    marker='.',
                    c=curr.sim_sizes(),
                    cmap='plasma')

        plt.show()

        #plt.savefig('ada_iter_{i}.png'.format(i=i))
        #plt.close()

    i += 1


n_pts = 0
s_max = 0
if not (curr is None):
    finals.append(curr)
for final in finals:
    n_pts += final.thetas().shape[1]
    if final.sim_sizes().size != 0:
        s_max = max(s_max, np.max(final.sim_sizes()))
print(n_pts)
print(s_max)
