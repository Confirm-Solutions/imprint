import pykevlar.core as core
import pykevlar.driver as driver
import numpy as np

n_arms = 3
ph2_size = 50
n_samples = 250
n_thetas = 100
sim_size = 10
seed = 69
thresh = 1.96

# set numpy random seed
np.random.seed(seed)

# create current batch of grid points
# Note that at the thread-level, we only need to know theta gridpoints.
# We technically don't need any valid radii.
# sim_sizes also are not read/written.
gr = core.GridRange(n_arms, n_thetas)
thetas = gr.get_thetas()
thetas[...] = np.random.normal(0., 1., size=thetas.shape)

# create BCKT
bckt = core.BinomialControlkTreatment(n_arms, ph2_size, n_samples, gr, thresh)
bckt_state = bckt.make_state()

# create RNG
gen = core.mt19937()
gen.seed(seed)

# run a mock-call of fit_thread
is_o = driver.fit_thread(bckt_state, sim_size, gen)
print(is_o.type_I_sum() / is_o.n_accum())
