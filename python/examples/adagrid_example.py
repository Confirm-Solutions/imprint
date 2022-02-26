from pykevlar.core import (
    GridRange,
    BinomialControlkTreatment,
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
#basicConfig(level = log_level,
#            format  = '%(asctime)s %(levelname)-8s %(module)-20s: %(message)s',
#            datefmt ='%Y-%m-%d %H:%M:%S')
logger = getLogger(__name__)

# ========== Toggleable ===============
n_arms = 2          # prioritize 2 first, then do 3, 4
max_iter = 15        # max iterations into adagrid
N_max = int(1E5)    # max simulation size
n_threads = os.cpu_count()
max_batch_size = 2000000

logger.info("n_arms: %d, max_iter: %d, N_max: %d, "
            "n_threads: %d, max_batch_size: %d" %
            (n_arms, max_iter,
             N_max, n_threads, max_batch_size))
# ========== End Toggleable ===============

init_sim_size = int(1E3) # initial simulation size
                    # (for simplicity, fixed for all points)
init_size = 8       # initial number of points along each direction
#lower = np.array([-1] * n_arms)  # lower bound for each direction
#upper = np.array([1] * n_arms)    # upper bound for each direction
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
#def null_hypo(i, p):
#    return p[i] <= p[0]
def null_hypo(p):
    return p[1] <= 1

# define is_not_alt
def is_not_alt(p):
    #return np.any(np.array([null_hypo(i, p) for i in range(1,n_arms)]))
    return p[1] <= 0

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
thetas = gr.get_thetas()
thetas[...] = np.transpose(grid)
radii = gr.get_radii()
for i, row in enumerate(radii):
    row[...] = radius[i]
sim_sizes = gr.get_sim_sizes()
sim_sizes[...] = init_sim_size

# create model
#model = BinomialControlkTreatment(n_arms, ph2_size, n_samples, [])
model = ExpControlkTreatment(n_samples, 2.0, [thr_minus, thr])
model.set_grid_range(gr, null_hypo)

is_o = fit_process(model=model, sim_size=init_sim_size, base_seed=seed)
print(is_o.type_I_sum())


# create batcher
batcher = SimpleBatch(max_size=max_batch_size)

adagrid = AdaGrid()
gr_new = adagrid.fit(
    batcher=batcher,
    model=model,
    null_hypo=null_hypo,
    is_not_alt=is_not_alt,
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
        thetas = curr.get_thetas_const()

        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.scatter(thetas[0,:], thetas[1,:], thetas[2,:],
        #            marker='.',
        #            c=curr.get_sim_sizes(),
        #            cmap='plasma')
        #ax.set_title('Iter={i}'.format(i=i))

        plt.scatter(thetas[0,:], thetas[1,:],
                    marker='.',
                    c=curr.get_sim_sizes(),
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
    n_pts += final.get_thetas().shape[1]
    if final.get_sim_sizes().size != 0:
        s_max = max(s_max, np.max(final.get_sim_sizes()))
print(n_pts)
print(s_max)
