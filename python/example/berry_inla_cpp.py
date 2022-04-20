import sys
sys.path.append('./research/berry/')
import util
import fast_inla
from scipy.special import logit
import matplotlib.pyplot as plt
import numpy as np
from pykevlar.core import mt19937
from pykevlar.grid import HyperPlane
from pykevlar.driver import accumulate_process

from utils import make_cartesian_grid_range

fi = fast_inla.FastINLA(2)

n_arms = 2
import pykevlar.core.model.binomial
model_type_name = 'BerryINLA' + str(n_arms)
model_type = getattr(pykevlar.core.model.binomial, model_type_name)
print(model_type)
seed = 10
n_theta_1d = 20
sim_size = 200
n_threads = 1

# define null hypos
null_hypos = []
for i in range(n_arms):
    n = np.zeros(n_arms)
    # null is:
    # theta_i <= logit(0.1)
    # the normal should point towards the negative direction. but that also
    # means we need to negate the logit(0.1) offset
    n[i] = -1
    null_hypos.append(HyperPlane(n, -logit(0.1)))

gr = make_cartesian_grid_range(n_theta_1d, np.full(n_arms, -6.5), np.full(n_arms, 0.0), sim_size)
gr.create_tiles(null_hypos)
plt.plot(gr.thetas()[0,:], gr.thetas()[1,:], 'ro')
gr.prune()
plt.plot(gr.thetas()[0,:], gr.thetas()[1,:], 'bo')
plt.show()

gr = make_cartesian_grid_range(n_theta_1d, np.full(n_arms, -6.5), np.full(n_arms, 0.0), sim_size)
gr.create_tiles(null_hypos)

y = np.array([[4, 5]])
n = np.array([[35, 35]])
b = model_type(
    n[0,0],
    [0.85], # final analysis exceedance requirement (note for interim analyss)
    np.full(2, fi.thresh_theta),
    fi.sigma2_rule.wts.copy(),
    fi.cov.reshape((-1, 4)).T.copy(),
    fi.neg_precQ.reshape((-1, 4)).T.copy(),
    fi.logprecQdet.copy(),
    fi.log_prior.copy(),
    fi.tol,
    fi.logit_p1
)
correct = fi.numpy_inference(y, n)[1][0]
exc = b.get_posterior_exceedance_probs(y[0])
np.testing.assert_allclose(exc, correct)

import time
start = time.time()
out = accumulate_process(b, gr, sim_size, seed, n_threads)
end = time.time()
print('runtime', end - start)
print(out.typeI_sum())
print(out.score_sum().reshape((-1, 2)))
pos_start = np.cumsum(gr.n_tiles_per_pt, dtype=np.int64) - gr.n_tiles_per_pt[0]
typeI_per_gridpt = np.add.reduceat(out.typeI_sum()[0], pos_start) / (sim_size * gr.n_tiles_per_pt[0])
plt.figure()
plt.scatter(gr.thetas()[0], gr.thetas()[1], c=typeI_per_gridpt)
plt.colorbar()
plt.show()
# gen = mt19937(seed)
# rej_len = np.arange(gr.n_tiles(), dtype=np.uint32)
# b.make_sim_global_state(gr).make_sim_state().simulate(gen, rej_len)

# TODO: 
# - replicate the 