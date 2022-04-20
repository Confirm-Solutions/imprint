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
import numpy as np
from pykevlar.grid import HyperPlane
from pykevlar.model.binomial import SimpleSelection
from pykevlar.core.model import mt19937
from pykevlar.driver import accumulate_process
from utils import make_cartesian_grid_range

n_arms = 2
n_arm_samples = 35
n_thetas = 3
sim_size = 100
seed = 10
model = SimpleSelection(n_arms, n_arm_samples, 20, [])
model.critical_values([2.1])

gr = make_cartesian_grid_range(n_thetas, np.full(n_arms, -0.5), np.full(n_arms, 0.5), 1000)

# define null hypos
null_hypos = []
for i in range(1, n_arms):
    n = np.zeros(n_arms)
    n[0] = 1
    n[i] = -1
    null_hypos.append(HyperPlane(n, 0))
gr.create_tiles(null_hypos)
gr.prune()

out = accumulate_process(model, gr, sim_size, seed, 1)

out.typeI_sum()
sgs = model.make_sim_global_state(gr)
ss = sgs.make_sim_state()
gen = mt19937(seed)
pos_start = np.cumsum(gr.n_tiles_per_pt, dtype=np.int64) - gr.n_tiles_per_pt[0]
typeI_sum = np.zeros(gr.n_tiles())
score_sum = np.zeros((gr.n_tiles(), 2))

# We can run in blocks of 1000 sim_size. Adagrid can just discretize in blocks
# of that size. Then, the set of grid points that need to be run over simulation
# block.
# for block in ...
rej_len = np.zeros((sim_size, gr.n_tiles()), dtype=np.uint32)
score = np.zeros((sim_size, gr.n_gridpts(), n_arms))
for i in range(sim_size):
    ss.simulate(gen, rej_len[i])
    rej_len_per_gridpt = np.add.reduceat(rej_len[i], pos_start)
    for j in np.where(rej_len_per_gridpt > 0)[0]:
        score_buf = np.empty(n_arms)
        ss.score(j, score_buf)
        score[i, j] += score_buf
    # ss.score(rej_idxs, score_buf)

typeI_sum = rej_len.sum(axis=0)
score_sum = score.sum(axis=0)
print(typeI_sum)
print(score_sum)
np.testing.assert_allclose(typeI_sum, out.typeI_sum()[0])
np.testing.assert_allclose(score_sum, out.score_sum().reshape((-1, 2)))

gen = mt19937(seed)
sample = np.array(gen.uniform_sample(n_arm_samples * n_arms)).reshape((n_arm_samples, n_arms));