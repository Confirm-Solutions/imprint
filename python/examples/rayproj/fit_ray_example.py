import numpy as np
import ray
import pykevlar.core as core
import pykevlar.driver as driver
    
ray.init(address='ray://yokojitsu:10001')

alive_nodes = [n for n in ray.nodes() if n['Alive']]
print('''This cluster consists of
    {} nodes in total
    {} CPU resources in total
'''.format(len(alive_nodes), ray.cluster_resources()['CPU']))

num_instances = 2

@ray.remote
def call_kevlar(bckt, sim_size):
    import sys
    sys.path.append('/home/ray/kevlar/python')
    import pykevlar.core as core
    import pykevlar.driver as driver
    
    # run a mock-call of fit_thread
    is_o = driver.fit_thread(bckt, sim_size, 69)
    return (is_o.type_I_sum() / is_o.n_accum())

# ========== Toggleable ===============
n_arms = 3      # prioritize 3 first, then do 4
sim_size = 100000
n_thetas_1d = 64
seed = 69
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
# At the process-level, we only need to know theta, radii.
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

# create BCKT
bckt = core.BinomialControlkTreatment(n_arms, ph2_size, n_samples, thresh)
bckt.set_grid_range(gr, null_hypo)

sim_size_thr = sim_size // num_instances
sim_size_rem = sim_size % num_instances

# create input arguments list
inputs = [
    (bckt,
     sim_size_thr + (i < sim_size_rem),
     seed + i)
    for i in range(num_instances)
]

object_ids = [call_kevlar.remote(i, sim_size) for i in inputs]
res = ray.get(object_ids)
for row in res:
    print(row)
