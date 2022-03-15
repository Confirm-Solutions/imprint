import pykevlar.core as core
import pykevlar.driver as driver
from pykevlar.batcher import SimpleBatch
import numpy as np
import os
from logging import basicConfig, getLogger
from logging import DEBUG as log_level
#from logging import INFO as log_level
basicConfig(level = log_level,
            format  = '%(asctime)s %(levelname)-8s %(module)-20s: %(message)s',
            datefmt ='%Y-%m-%d %H:%M:%S')
logger = getLogger(__name__)

# ========== Toggleable ===============
n_arms = 3      # prioritize 3 first, then do 4
sim_size = 1E5
n_thetas_1d = 64
n_threads = os.cpu_count()
max_batch_size = -1

logger.info("n_arms: %d, sim_size %d, n_thetas_1d: "
            "%d, n_threads: %d, max_batch_size: %d" %
            (n_arms, sim_size, n_thetas_1d,
             n_threads, max_batch_size))
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
null_hypos = []
for i in range(1, n_arms):
    n = np.zeros(n_arms)
    n[0] = 1
    n[i] = -1
    null_hypos.append(core.HyperPlane(n, 0))

# Create full grid.
# At the driver-level, we need to know theta, radii, sim_sizes.
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

# create batcher
batcher = SimpleBatch(gr, max_batch_size, null_hypos)

# create BCKT
bckt = core.BinomialControlkTreatment(n_arms, ph2_size, n_samples, [thresh])

# run a mock-call of fit_driver
# Currently, it will yield each batched result.
# TODO: once this doesn't yield anymore, modify this part.
for is_o in driver.fit_driver(batcher, bckt, seed, n_threads):
    logger.info(is_o.type_I_sum() / sim_size)
