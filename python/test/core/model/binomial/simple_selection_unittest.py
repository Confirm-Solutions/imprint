import unittest

import numpy as np
import pyimprint.core.grid as grid
import pyimprint.core.model.binomial as binom


class TestSimpleSelection(unittest.TestCase):
    def make_model(self, n_arms, n_arm_samples, n_phase2_samples, critical_values):
        return binom.SimpleSelection(
            n_arms=n_arms,
            n_arm_samples=n_arm_samples,
            n_phase2_samples=n_phase2_samples,
            critical_values=critical_values,
        )

    def make_grid_range(self, n_params, n_gridpts):
        return grid.GridRange(n_params, n_gridpts)

    def test_constructor(self):
        self.make_model(3, 10, 5, [3])

    def test_n_arms(self):
        m = self.make_model(3, 10, 5, [3])
        self.assertEqual(m.n_arms(), 3)

    def test_n_arm_samples(self):
        m = self.make_model(3, 10, 5, [3])
        self.assertEqual(m.n_arm_samples(), 10)

    def test_n_phase2_samples(self):
        m = self.make_model(3, 10, 5, [3])
        self.assertEqual(m.n_phase2_samples(), 5)

    def test_critical_values(self):
        m = self.make_model(3, 10, 5, [3])
        self.assertTrue((m.critical_values() == np.array([3])).all())

        m.critical_values([2, 5])
        self.assertTrue((m.critical_values() == np.array([5, 2])).all())

    def test_make_state(self):
        m = self.make_model(3, 10, 5, [3])

        # the following just needs to run without error
        gr = self.make_grid_range(3, 5)
        sgs = m.make_sim_global_state(gr)
        sgs.make_sim_state(0)

        m.make_imprint_bound_state(gr)
