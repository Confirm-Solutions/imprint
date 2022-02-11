import pykevlar.core as core
import pykevlar.driver as driver
import numpy as np
import os
import timeit

n_arms = 3
ph2_size = 50
n_samples = 250
sim_size = 100000
n_thetas_1d = 16
n_thetas = int(n_thetas_1d**n_arms)
seed = 69
thresh = 1.96
n_threads = os.cpu_count()
lower = -0.5
upper = 0.5

# set numpy random seed
np.random.seed(seed)

# create current batch of grid points
# Note that at the thread-level, we only need to know theta gridpoints.
# We technically don't need any valid radii.
# sim_sizes also are not read/written.
gr = core.GridRange(n_arms, n_thetas)
thetas = gr.get_thetas()
theta_1d = core.Gridder.make_grid(n_thetas_1d, lower, upper)
thetas[...] = np.transpose(
    np.stack(np.meshgrid(*(theta_1d for _ in range(n_arms))), axis=-1) \
        .reshape(-1, n_arms)
)

# create BCKT
bckt = core.BinomialControlkTreatment(n_arms, ph2_size, n_samples, gr, thresh)

# run a mock-call of fit_thread
print(timeit.timeit('driver.fit_process(bckt, sim_size, seed, n_threads)', number=1, globals=globals()))
#print(is_o.type_I_sum() / is_o.n_accum())
