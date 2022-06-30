import numpy as np
import pyimprint.grid as grid
import scipy.special
from pyimprint.core.model import ModelBase, SimStateBase


class SimState(SimStateBase):
    def __init__(self, outer, seed):
        SimStateBase.__init__(self)
        self.outer = outer
        self.n_arm_samples = self.outer.outer.n_arm_samples
        self.fast_inla_obj = self.outer.outer.fast_inla_obj
        self.p_tiles = scipy.special.expit(self.outer.theta_tiles)
        self.p = scipy.special.expit(self.outer.theta)
        np.random.seed(seed)

    def simulate(self, rej_len):
        self.uniform_samples = np.random.uniform(
            size=(self.n_arm_samples, self.fast_inla_obj.n_arms)
        )
        y = np.sum(self.uniform_samples[None] < self.p_tiles[:, None, :], axis=1)
        n = np.full_like(y, self.n_arm_samples)
        did_reject = self.fast_inla_obj.rejection_inference(np.stack((y, n), axis=-1))

        rej_len[...] = np.any(self.outer.nulls & did_reject, axis=-1)

    def score(self, gridpt_idx, out):
        y = np.sum(self.uniform_samples < self.p[gridpt_idx, None, :], axis=0)
        out[...] = y - self.n_arm_samples * self.p[gridpt_idx]


class SimGlobalState:
    def __init__(self, outer, gr):
        self.outer = outer
        self.theta = gr.thetas().T
        self.theta_tiles = grid.theta_tiles(gr)
        self.nulls = grid.is_null_per_arm(gr)

    def make_sim_state(self, seed):
        return SimState(self, seed)


class BerryImprintModel(ModelBase):
    def __init__(self, fast_inla_obj, n_arm_samples, cvs):
        """
        cvs:    critical values (descending order)
        """
        ModelBase.__init__(self, cvs)
        self.fast_inla_obj = fast_inla_obj
        self.n_arm_samples = n_arm_samples

    def make_sim_global_state(self, gr):
        return SimGlobalState(self, gr)
