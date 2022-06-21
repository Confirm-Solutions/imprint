import unittest

import numpy as np
from pyimprint.core.grid import GridRange


class TestGridRange(unittest.TestCase):
    def test_constructor(self):
        gr = GridRange(2, 3)
        self.assertEqual(gr.n_params(), 2)
        self.assertEqual(gr.n_gridpts(), 3)
        self.assertEqual(gr.n_tiles(), 0)

    def test_constructor_sugar(self):
        thetas = np.zeros((3, 2))
        radii = np.ones(thetas.shape)
        sim_sizes = 100 * np.ones(thetas.shape[1])
        gr = GridRange(thetas, radii, sim_sizes)
        self.assertTrue((thetas == gr.thetas()).all())
        self.assertTrue((radii == gr.radii()).all())
        self.assertTrue((sim_sizes == gr.sim_sizes()).all())
