import unittest

from core_test.model import test_py_sgs, test_py_ss_score, test_py_ss_simulate
from pykevlar.core.model import SimGlobalStateBase, SimStateBase


class PySS(SimStateBase):
    def simulate(self, gen, rej_len):
        rej_len[...] = 3

    def score(self, gridpt_idx, out):
        out[...] = 2.1


class PySGS(SimGlobalStateBase):
    def make_sim_state(self):
        return PySS()


class TestBase(unittest.TestCase):
    def make_py_sgs(self):
        return PySGS()

    def test_py_sgs(self):
        sgs = self.make_py_sgs()
        ss = test_py_sgs(sgs)
        self.assertTrue(isinstance(ss, SimStateBase))

    def test_py_ss_simulate(self):
        sgs = self.make_py_sgs()
        ss = sgs.make_sim_state()
        rej_len = test_py_ss_simulate(ss)
        self.assertTrue((rej_len == 3).all())

    def test_py_ss_score(self):
        sgs = self.make_py_sgs()
        ss = sgs.make_sim_state()
        out = test_py_ss_score(ss)
        self.assertTrue((out == 2.1).all())
