from logging import basicConfig, getLogger
from logging import DEBUG as log_level
#from logging import INFO as log_level
basicConfig(level = log_level,
            format  = '%(asctime)s %(levelname)-8s %(module)-20s: %(message)s',
            datefmt ='%Y-%m-%d %H:%M:%S')
logger = getLogger(__name__)

import pykevlar.core as core
import pykevlar.driver as driver
from pykevlar.batcher import SimpleBatch

import numpy as np
# import os
# from timeit import default_timer as timer
# from datetime import timedelta

import ray

ray.init(address='ray://yokojitsu:10001')
alive_nodes = [n for n in ray.nodes() if n['Alive']]
logger.info("ray available nodes: %d" % len(alive_nodes))
logger.info("ray available resources: %d" % ray.cluster_resources()['CPU'])

# ========== Toggleable ===============
n_arms = 4      # prioritize 3 first, then do 4
sim_size = 1000000
n_thetas_1d = 64
n_threads = 4 # os.cpu_count()

# ========== End Toggleable ===============

ph2_size = 50
n_samples = 250
seed = 69
thresh = 1.96
lower = -0.5
upper = 0.5

logger.info("n_arms: %d" % n_arms)
logger.info("sim_size: %d" % sim_size)
logger.info("n_thetas_1d: %d" % n_thetas_1d)
logger.info("n_threads: %d" % n_threads)

# set numpy random seed
np.random.seed(seed)

@ray.remote
def call_kevlar(batch, sim_size):
    return driver.fit_process(model=bckt,
                              grid_range=batch,
                              sim_size=sim_size,
                              base_seed=seed,
                              n_threads=n_threads)

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

max_batch_size = int(gr.n_gridpts() / ray.cluster_resources()['CPU']) + 1
logger.info("max_batch_size: %d" % max_batch_size)

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
    
# Call kevlar via ray remote
ray_objs = []
for batch, sim_size in batcher:
    ray_objs.append(call_kevlar.remote(batch, sim_size))
res = ray.get(ray_objs)

# batch_num = 0
for is_o in res:
    logger.info(is_o.type_I_sum() / sim_size)
    # batch_num = batch_num + 1
    # np.savetxt("ray_output_%05d.txt" % batch_num, (is_o.type_I_sum() / sim_size), delimiter="\n")