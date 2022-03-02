# from collections import Counter
import numpy as np
# import socket
# import time

# import os
import ray
ray.init(address='ray://yokojitsu:10001')


print('''This cluster consists of
    {} nodes in total
    {} CPU resources in total
'''.format(len(ray.nodes()), ray.cluster_resources()['CPU']))

num_instances = 2
@ray.remote
def call_kevlar():
    import sys
    
    sys.path.append('/home/ray/kevlar/python')
    import pykevlar.core as core
    import pykevlar.driver as driver
    from pykevlar.batcher import SimpleBatch
    # import numpy as np
    #import os
    #import timeit
    
    # res = ", ".join(list(os.environ))
    # from logging import basicConfig, getLogger
    # from logging import DEBUG as log_level
    # #from logging import INFO as log_level
    # basicConfig(level = log_level,
    #             format  = '%(asctime)s %(levelname)-8s %(module)-20s: %(message)s',
    #             datefmt ='%Y-%m-%d %H:%M:%S')
    # logger = getLogger(__name__)
    
    # ========== Toggleable ===============
    n_arms = 3      # prioritize 3 first, then do 4
    sim_size = 1E5
    n_thetas_1d = 64
    n_threads = 8
    max_batch_size = 64000
    
    # logger.info("n_arms: %d, sim_size %d, n_thetas_1d: %d, n_threads: %d, max_batch_size: %d" % (n_arms, sim_size, n_thetas_1d, n_threads, max_batch_size))
    # ========== End Toggleable ===============
    
    ph2_size = 50
    n_samples = 250
    seed = 69
    thresh = np.array([np.float64(1.96)])
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
    print(n_arms, ph2_size, n_samples, thresh)
    bckt = core.BinomialControlkTreatment(n_arms, ph2_size, n_samples, thresh)
    
    # run a mock-call of fit_driver
    # Currently, it will yield each batched result.
    # TODO: once this doesn't yield anymore, modify this part.
    res = []
    for is_o in driver.fit_driver(batcher, null_hypo, bckt, seed, n_threads):
        type_I_sum = is_o.type_I_sum()
        n_accum = is_o.n_accum()
        res.append(type_I_sum / n_accum)
    return res

object_ids = [call_kevlar.remote() for _ in range(num_instances)]
res = ray.get(object_ids)
for row in res:
    print(row)
# print(np.array_str(res[0], precision=8))
    
# for is_o in driver.fit_driver(batcher, null_hypo, bckt, seed, n_threads):
#     logger.info(is_o.type_I_sum() / is_o.n_accum())
