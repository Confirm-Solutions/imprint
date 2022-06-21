import unittest

from core_test.model import test_py_ss_score, test_py_ss_simulate
from pyimprint.core.model import SimStateBase


class PySS(SimStateBase):
    def __init__(self, seed):
        SimStateBase.__init__(self)

    def simulate(self, rej_len):
        rej_len[...] = 3

    def score(self, gridpt_idx, out):
        out[...] = 2.1


class PySGS:
    def make_sim_state(self, seed):
        return PySS(seed)


class TestBase(unittest.TestCase):
    def make_py_sgs(self):
        return PySGS()

    def test_py_ss_simulate(self):
        sgs = self.make_py_sgs()
        ss = sgs.make_sim_state(0)
        rej_len = test_py_ss_simulate(ss)
        self.assertTrue((rej_len == 3).all())

    def test_py_ss_score(self):
        sgs = self.make_py_sgs()
        ss = sgs.make_sim_state(0)
        out = test_py_ss_score(ss)
        self.assertTrue((out == 2.1).all())
