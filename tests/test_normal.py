import numpy as np

import imprint.bound.normal as normal


def fwd_qcp_derivative(q, scale, v, f0):
    return 0.5 * (scale * v) ** 2 + np.log(f0) / q**2


def bwd_qcp_derivative(q, scale, v, alpha):
    return 0.5 * (scale * v) ** 2 + np.log(alpha) / (q - 1) ** 2


def tile_fwd_qcp_derivative(q, scale, vs, f0):
    mv = np.max((scale * vs) ** 2)
    return 0.5 * mv + np.log(f0) / q**2


def tile_bwd_qcp_derivative(q, scale, vs, alpha):
    mv = np.max((scale * vs) ** 2)
    return 0.5 * mv + np.log(alpha) / (q - 1) ** 2


def test_fwd_qcp_solver():
    scale = 2.0
    v = -0.321
    f0 = 0.025
    fwd_solver = normal.ForwardQCPSolver(scale)
    q_opt = fwd_solver.solve(v, f0)
    q_opt_deriv = fwd_qcp_derivative(q_opt, scale, v, f0)
    np.testing.assert_almost_equal(q_opt_deriv, 0.0)


def test_fwd_qcp_solver_inf():
    scale = 2.0
    v = 0
    f0 = 0.025
    fwd_solver = normal.ForwardQCPSolver(scale)
    q_opt = fwd_solver.solve(v, f0)
    np.testing.assert_almost_equal(q_opt, np.inf)


def test_bwd_qcp_solver():
    scale = 2.0
    v = -0.321
    alpha = 0.025
    bwd_solver = normal.BackwardQCPSolver(scale)
    q_opt = bwd_solver.solve(v, alpha)
    q_opt_deriv = bwd_qcp_derivative(q_opt, scale, v, alpha)
    np.testing.assert_almost_equal(q_opt_deriv, 0.0)


def test_tile_fwd_qcp_solver():
    scale = 3.2
    vs = np.array([-0.1, 0.2])
    f0 = 0.025
    fwd_solver = normal.TileForwardQCPSolver(scale)
    q_opt = fwd_solver.solve(vs, f0)
    q_opt_deriv = tile_fwd_qcp_derivative(q_opt, scale, vs, f0)
    np.testing.assert_almost_equal(q_opt_deriv, 0.0)


def test_tile_bwd_qcp_solver():
    scale = 1.2
    vs = np.array([-0.3, 0.1])
    alpha = 0.025
    bwd_solver = normal.TileBackwardQCPSolver(scale)
    q_opt = bwd_solver.solve(vs, alpha)
    q_opt_deriv = tile_bwd_qcp_derivative(q_opt, scale, vs, alpha)
    np.testing.assert_almost_equal(q_opt_deriv, 0.0)


def test_fwd_bwd_invariance():
    scale = 2.0
    v = -0.321
    f0 = 0.025
    q = 3.2
    fwd_bound = normal.tilt_bound_fwd(q, scale, v, f0)
    bwd_bound = normal.tilt_bound_bwd(q, scale, v, fwd_bound)
    np.testing.assert_almost_equal(bwd_bound, f0)


def test_tile_fwd_bwd_invariance():
    scale = 1.2
    vs = np.array([-0.3, 0.1])
    f0 = 0.025
    q = 5.1
    fwd_bound = normal.tilt_bound_fwd_tile(q, scale, vs, f0)
    bwd_bound = normal.tilt_bound_bwd_tile(q, scale, vs, fwd_bound)
    np.testing.assert_almost_equal(bwd_bound, f0)
