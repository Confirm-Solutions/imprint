import numpy as np
from pyimprint.core.model import ModelBase, SimStateBase


class SimState(SimStateBase):
    def __init__(self, outer, seed):
        SimStateBase.__init__(self)
        self.outer = outer
        self.std_normal = 0
        np.random.seed(seed)

    def simulate__(mus, comp, nulls):
        return nulls & (mus[0, :] > comp)

    def simulate(self, rej_len):
        cvs = self.outer.outer.critical_values()

        self.std_normal = np.random.normal()

        rej_len[...] = SimState.simulate__(
            self.outer.thetas,
            cvs[0] - self.std_normal,
            self.outer.nulls,
        )

    def score(self, gridpt_idx, out):
        out[...] = self.std_normal


class SimGlobalState:
    def __init__(self, outer, gr):
        self.outer = outer
        self.thetas = gr.thetas()
        self.nulls = np.array(
            [
                gr.check_null(i, j, 0)
                for i in range(gr.n_gridpts())
                for j in range(gr.n_tiles(i))
            ]
        )

    def make_sim_state(self, seed):
        return SimState(self, seed)


class Simple(ModelBase):
    def __init__(self, cvs):
        """
        cvs:    critical values (descending order)
        """
        self.n_arms = 1
        self.n_arm_samples = 1
        ModelBase.__init__(self, cvs)

    def make_sim_global_state(self, gr):
        return SimGlobalState(self, gr)
