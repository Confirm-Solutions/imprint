import numpy as np
from pykevlar.core.model import mt19937
from pykevlar.driver import accumulate_process
from pykevlar.grid import HyperPlane
from pykevlar.model.binomial import SimpleSelection
from utils import make_cartesian_grid_range

n_arms = 2
n_thetas = 3
sim_size = 100
seed = 10


def model_gr():
    model = SimpleSelection(n_arms, 35, 20, [])
    model.critical_values([2.1])

    gr = make_cartesian_grid_range(
        n_thetas, np.full(n_arms, -0.5), np.full(n_arms, 0.5), 1000
    )

    # define null hypos
    null_hypos = []
    for i in range(1, n_arms):
        n = np.zeros(n_arms)
        n[0] = 1
        n[i] = -1
        null_hypos.append(HyperPlane(n, 0))
    gr.create_tiles(null_hypos)
    gr.prune()
    return model, gr


def test_compare():
    model, gr = model_gr()
    out = accumulate_process(model, gr, sim_size, seed, 1)
    print(out.typeI_sum())
    print(out.score_sum())


class CombinedModel:
    def __init__(self, gr, sim_size, seed):
        self.gr = gr
        self.sim_size = sim_size
        self.seed = seed


def test_score():
    model, gr = model_gr()
    sgs = model.make_sim_global_state(gr)
    ss = sgs.make_sim_state()
    gen = mt19937(seed)
    rej_len = np.empty(gr.n_tiles(), dtype=np.uint32)
    ss.simulate(gen, rej_len)
    score_buf = np.empty(n_arms)
    ss.score(0, score_buf)
    print(score_buf)


def test_compare2():
    model, gr = model_gr()
    sgs = model.make_sim_global_state(gr)
    ss = sgs.make_sim_state()
    gen = mt19937(seed)
    rej_len = np.empty(gr.n_tiles(), dtype=np.uint32)
    ss.simulate(gen, rej_len)
    # typeI_sum = np.empty(gr.n_tiles())
    # score_sum = np.zeros(gr.n_tiles())

    score = np.empty((gr.n_gridpts(), n_arms))
    for i in range(gr.n_gridpts()):
        score_buf = np.empty(n_arms)
        ss.score(i, score_buf)
        score[i] = score_buf
    print(score)
    pos_start = np.cumsum(gr.n_tiles_per_pt, dtype=np.int64) - gr.n_tiles_per_pt[0]
    print(pos_start.dtype, pos_start.shape)
    # print(rej_len[(pos_start[:-1]):(pos_start[1:])])
    print(pos_start)
    # np.where(gr.n_tiles_per_pt)
