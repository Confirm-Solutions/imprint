import numpy as np
import ray
# import sys
# sys.path.append('/home/mulderg/Work/kevlar/python')
from pykevlar.core import Gridder, GridRange, BinomialControlkTreatment
from pykevlar.driver import fit_thread
# import pykevlar.core as core
# import pykevlar.driver as driver
    
ray.init(address='ray://yokojitsu:10001')

# alive_nodes = [n for n in ray.nodes() if n['Alive']]
# print('''This cluster consists of
#     {} nodes in total
#     {} CPU resources in total
# '''.format(len(alive_nodes), ray.cluster_resources()['CPU']))

@ray.remote
def call_kevlar(params):
    # import sys
    # sys.path.append('/home/ray/kevlar/python')
    # from pykevlar.driver import fit_thread
    # def fit_thread(a, b, c):
    #     return True
    return fit_thread(params[0], params[1], params[3])

n_instances = 2

# ========== Toggleable ===============
n_arms = 3      # prioritize 3 first, then do 4
sim_size = 10000
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
theta_1d = Gridder.make_grid(n_thetas_1d, lower, upper)
grid = np.stack(np.meshgrid(*(theta_1d for _ in range(n_arms))), axis=-1) \
        .reshape(-1, n_arms)
grid_null = np.array([
    p for p in grid if null_hypo(1, p) or null_hypo(2, p)
])
gr = GridRange(n_arms, grid_null.shape[0])
thetas = gr.get_thetas()
thetas[...] = np.transpose(grid_null)

# create BCKT
bckt = BinomialControlkTreatment(n_arms, ph2_size, n_samples, [thresh])
bckt.set_grid_range(gr, null_hypo)

sim_size_thr = sim_size // n_instances
sim_size_rem = sim_size % n_instances

# The following thread logic should change to something more clever.
# ========= THREAD LOGIC ============

# create input arguments list
inputs = [
    (bckt,
     sim_size_thr + (i < sim_size_rem),
     seed + i)
    for i in range(n_instances)
]
    
# run a mock-call of fit_thread
# is_o = driver.fit_thread(bckt, sim_size, seed)
# print(is_o.type_I_sum() / sim_size)

object_ids = [call_kevlar.remote(i) for i in inputs]
res = ray.get(object_ids)
for row in res:
    print(row.type_I_sum() / sim_size)
