from logging import basicConfig, getLogger
from logging import DEBUG as log_level
#from logging import INFO as log_level
basicConfig(level = log_level,
            format  = '%(asctime)s %(levelname)-8s %(module)-20s: %(message)s',
            datefmt ='%Y-%m-%d %H:%M:%S')
logger = getLogger(__name__)

import pykevlar.core as core
import pykevlar.driver as driver
import numpy as np
# import os
from timeit import default_timer as timer
from datetime import timedelta

import ray

ray.init(address='ray://yokojitsu:10001')
alive_nodes = [n for n in ray.nodes() if n['Alive']]
logger.info("ray number of nodes: %d" % len(alive_nodes))
logger.info("ray number of resources: %d" % ray.cluster_resources()['CPU'])

# ========== Toggleable ===============

n_arms = 3      # prioritize 3 first, then do 4
sim_size = 100000
n_thetas_1d = 64
n_threads = 8 #os.cpu_count()

# ray nodes used
n_ray_nodes = 1

# ========== End Toggleable ===============

ph2_size = 50
n_samples = 250
seed = 69
thresh = 2.1
lower = -0.5
upper = 0.5

logger.info("n_arms: %d" % n_arms)
logger.info("sim_size: %d" % sim_size)
logger.info("n_thetas_1d: %d" % n_thetas_1d)
logger.info("n_threads: %d" % n_threads)
logger.info("n_ray_nodes: %d" % n_ray_nodes)

# set numpy random seed
np.random.seed(seed)

@ray.remote
def call_kevlar(i):
    
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
    
    gr.create_tiles(null_hypos)
    
    start = timer()
    gr.prune()
    end = timer()
    logger.info("Prune time: {t}".format(t=timedelta(seconds=end-start)))
    
    logger.info("Gridpts: {n}".format(n=gr.n_gridpts()))
    logger.info("Tiles: {n}".format(n=gr.n_tiles()))
    
    # create BCKT
    bckt = core.BinomialControlkTreatment(n_arms, ph2_size, n_samples, [thresh])
    
    # run a mock-call of fit_process
    start = timer()
    is_o = driver.fit_process(bckt, gr, sim_size, seed, n_threads)
    end = timer()
    logger.info("Fit time: {t}".format(t=timedelta(seconds=end-start)))
    logger.info((is_o.type_I_sum() / sim_size))

# Call kevlar via ray remote n_ray_nodes times
object_ids = [call_kevlar.remote(i) for i in range(n_ray_nodes)]
res = ray.get(object_ids)
for row in res:
    logger.info(row)