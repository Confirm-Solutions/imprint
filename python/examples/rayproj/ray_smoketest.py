# from collections import Counter
import numpy as np
# import socket
# import time
# import pykevlar.core as core
# import pykevlar.driver as driver
    
# import os
import ray
ray.init(address='ray://yokojitsu:10001')

alive_nodes = [n for n in ray.nodes() if n['Alive']]
print('''This cluster consists of
    {} nodes in total
    {} CPU resources in total
'''.format(len(alive_nodes), ray.cluster_resources()['CPU']))

num_instances = 1
@ray.remote
def call_kevlar(i):
    import sys
    sys.path.append('/home/ray/kevlar/python')
    import pykevlar.core as core
    import pykevlar.driver as driver
    
    # ========== Toggleable ===============
    n_arms = 3      # prioritize 3 first, then do 4
    sim_size = 100000
    n_thetas_1d = 64
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
    
    # Create current batch of grid points.
    # At the thread-level, we only need to know theta gridpoints.
    theta_1d = core.Gridder.make_grid(n_thetas_1d, lower, upper)
    grid = np.stack(np.meshgrid(*(theta_1d for _ in range(n_arms))), axis=-1) \
            .reshape(-1, n_arms)
    grid_null = np.array([
        p for p in grid if null_hypo(1, p) or null_hypo(2, p)
    ])
    gr = core.GridRange(n_arms, grid_null.shape[0])
    thetas = gr.get_thetas()
    thetas[...] = np.transpose(grid_null)
    
    # create BCKT
    bckt = core.BinomialControlkTreatment(n_arms, ph2_size, n_samples, thresh)
    bckt.set_grid_range(gr, null_hypo)
    
    # run a mock-call of fit_thread
    is_o = driver.fit_thread(bckt, sim_size, seed)
    return (i, (is_o.type_I_sum() / is_o.n_accum()))


object_ids = [call_kevlar.remote(i) for i in range(num_instances)]
res = ray.get(object_ids)
for row in res:
    print(row)
