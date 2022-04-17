import sys
sys.path.append('./research/berry/')
import util
import fast_inla
import numpy as np
from pykevlar.core.model.binomial import BerryINLA
from pykevlar.core.model import mt19937
from pykevlar.grid import HyperPlane

from utils import make_cartesian_grid_range

# fi = fast_inla.FastINLA()

n_arms = 2
seed = 10

# define null hypos
null_hypos = []
for i in range(1, n_arms):
    n = np.zeros(n_arms)
    n[0] = 1
    n[i] = -1
    null_hypos.append(HyperPlane(n, 0))


gr = make_cartesian_grid_range(3, np.full(n_arms, -0.5), np.full(n_arms, 0.5), 1000)
gr.create_tiles(null_hypos)
gr.prune()

b = BerryINLA(
    n_arms, 25,
    [0.0],
    [0.95, 0.95],
    fi.sigma2_rule.pts,
    fi.sigma2_rule.wts,
    fi.cov.reshape((-1, 4)),
    fi.neg_precQ.reshape((-1, 4)),
    fi.logprecQdet,
    fi.log_prior,
    fi.tol
)

gen = mt19937(seed)
rej_len = np.empty(gr.n_tiles(), dtype=np.uint32)
b.make_sim_global_state(gr).make_sim_state().simulate(gen, rej_len)