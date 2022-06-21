import unittest

import numpy as np
from pyimprint.driver import accumulate_process
from pyimprint.grid import Gridder, GridRange
from pyimprint.model.normal import Simple


class TestSimple(unittest.TestCase):
    def make_model(self, cvs):
        return Simple(cvs)

    def make_grid_range(self, n, lower, upper):
        thetas = Gridder.make_grid(n, lower, upper).reshape((1, n))
        radii = Gridder.radius(n, lower, upper) * np.ones((1, n))
        sim_sizes = 10 * np.ones(n, dtype=np.uint32)
        gr = GridRange(thetas, radii, sim_sizes, [])
        return gr

    def test_make_model(self):
        model = self.make_model([1.96])
        self.assertTrue(model.critical_values()[0] == 1.96)

    def test_make_sim_global_state(self):
        model = self.make_model([1.96])
        gr = self.make_grid_range(10, -3, 0)
        sgs = model.make_sim_global_state(gr)
        assert sgs

    def test_ss_simulate(self):
        from core_test.model import test_py_ss_simulate

        model = self.make_model([1.96])
        gr = self.make_grid_range(10, -3, 0)
        sgs = model.make_sim_global_state(gr)
        ss = sgs.make_sim_state(0)
        out = test_py_ss_simulate(ss)
        self.assertTrue((out == 0).all())

    def test_ss_score(self):
        from core_test.model import test_py_ss_score

        model = self.make_model([1.96])
        gr = self.make_grid_range(10, -3, 0)
        sgs = model.make_sim_global_state(gr)
        ss = sgs.make_sim_state(0)
        out = test_py_ss_score(ss)
        self.assertTrue((out == 0).all())

    def test_example(self):
        lower = -3.0
        upper = 1.4
        cv = 1.96
        model = self.make_model([upper + cv])
        gr = self.make_grid_range(10, lower, upper)
        sim_size = int(1e3)
        acc_o = accumulate_process(model, gr, sim_size=sim_size, base_seed=0)
        print(acc_o.typeI_sum() / sim_size)
