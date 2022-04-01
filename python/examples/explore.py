# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3.10.2 ('kevlar')
#     language: python
#     name: python3
# ---

# +
import pykevlar.core as core
import pykevlar.driver as driver
from pykevlar.batcher import SimpleBatch
import numpy as np
import os
#import timeit

from logging import basicConfig, getLogger
# from logging import DEBUG as log_level
from logging import INFO as log_level
basicConfig(level = log_level,
            format  = '%(asctime)s %(levelname)-8s %(module)-20s: %(message)s',
            datefmt ='%Y-%m-%d %H:%M:%S')
logger = getLogger(__name__)


# +

# ========== Toggleable ===============
n_arms = 2      # prioritize 3 first, then do 4
sim_size = 1E5
n_thetas_1d = 64
n_threads = os.cpu_count()
max_batch_size = 64000

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
def null_hypo(i, p):
    return p[i] <= p[0]


# +


# Create full grid.
# At the driver-level, we need to know theta, radii, sim_sizes.
theta_1d = core.Gridder.make_grid(n_thetas_1d, lower, upper)
grid = np.stack(np.meshgrid(*(theta_1d for _ in range(n_arms))), axis=-1) \
        .reshape(-1, n_arms)

grid_null = np.array([
    p for p in grid if null_hypo(1, p)
])

grid_null.shape, grid.shape

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
bckt = core.BinomialControlkTreatment(n_arms, ph2_size, n_samples, [thresh])
bckt.set_grid_range(gr, null_hypo)
bckt.cov_quad(0, radii[:, 0])

# +

# is_o = driver.fit_thread(bckt, sim_size, seed)
# print(is_o.type_I_sum() / sim_size)


# +
class MyBinomialState:
    def __init__(self, model):
        self.model = model

    def gen_rng(self, gen):
        pass

    def gen_suff_stat(self):
        pass

class MyBinomial23(core.BinomialControlkTreatment):
    def __init__(self, n_arms, ph2_size, n_samples, thresholds):
        super().__init__(n_arms, ph2_size, n_samples, thresholds)

    def make_state(self):
        return MyBinomialState(self)
