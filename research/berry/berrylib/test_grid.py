import time

import numpy as np
import pytest
from berrylib.grid import HyperPlane, build_grid, get_edges, prune
from numpy import nan


def normalize(n):
    return n / np.linalg.norm(n)


@pytest.fixture
def simple_grid():
    thetas = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    radii = np.full_like(thetas, 0.5)
    hypos = [
        HyperPlane(normalize(np.array([1, -1])), 0),
        HyperPlane(normalize(np.array([1, 1])), -1),
    ]
    return build_grid(thetas, radii, hypos)


def test_edge_vecs():
    edges = get_edges(np.array([[1, 0]]), np.array([[1, 2]]))
    correct = np.array([[[2, -2, 0, 4], [0, 2, 2, 0], [0, -2, 2, 0], [0, -2, 0, 4]]])
    np.testing.assert_allclose(edges, correct)


def test_tile_split(simple_grid):
    g = simple_grid
    np.testing.assert_allclose(g.grid_pt_idx, [0, 0, 0, 0, 1, 2, 3, 3])
    np.testing.assert_allclose(g.is_regular, [0, 0, 0, 0, 1, 1, 0, 0])
    np.testing.assert_allclose(
        g.null_truth,
        np.array([[1, 1], [1, 0], [0, 1], [0, 0], [0, 1], [1, 1], [1, 1], [0, 1]]),
    )
    np.testing.assert_allclose(
        g.vertices,
        np.array(
            [
                [[-1.0, -1.0], [0.0, -1.0], [0.0, 0.0], [nan, nan]],
                [[-1.0, -1.0], [0.0, -1.0], [0.0, 0.0], [nan, nan]],
                [[-1.0, -1.0], [-1.0, 0.0], [0.0, 0.0], [nan, nan]],
                [[-1.0, -1.0], [-1.0, 0.0], [0.0, 0.0], [nan, nan]],
                [[-1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, -1.0], [0.0, -1.0], [1.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [nan, nan]],
                [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [nan, nan]],
            ]
        ),
    )


def test_tile_prune(simple_grid):
    g = simple_grid
    gp = prune(g)
    np.testing.assert_allclose(gp.grid_pt_idx, [0, 0, 0, 1, 2, 3, 3])
    np.testing.assert_allclose(gp.is_regular, [0, 0, 0, 1, 1, 0, 0])
    np.testing.assert_allclose(
        gp.null_truth,
        np.array([[1, 1], [1, 0], [0, 1], [0, 1], [1, 1], [1, 1], [0, 1]]),
    )
    np.testing.assert_allclose(
        gp.vertices,
        np.array(
            [
                [[-1.0, -1.0], [0.0, -1.0], [0.0, 0.0], [nan, nan]],
                [[-1.0, -1.0], [0.0, -1.0], [0.0, 0.0], [nan, nan]],
                [[-1.0, -1.0], [-1.0, 0.0], [0.0, 0.0], [nan, nan]],
                [[-1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, -1.0], [0.0, -1.0], [1.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [nan, nan]],
                [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [nan, nan]],
            ]
        ),
    )


def test_prune_off_gridpt():
    thetas = np.array([[-0.5, -0.5], [0.5, 0.5]])
    radii = np.full_like(thetas, 0.5)
    hypos = [HyperPlane(normalize(np.array([1, 1])), 0)]
    g = prune(build_grid(thetas, radii, hypos))
    np.testing.assert_allclose(g.thetas, np.array([[0.5, 0.5]]))
    np.testing.assert_allclose(g.grid_pt_idx, np.array([0]))


def test_prune_is_regular():
    thetas = np.array([[0.0, 0.0]])
    radii = np.full_like(thetas, 0.5)
    hypos = [HyperPlane(normalize(np.array([1, 1])), 0)]
    g = build_grid(thetas, radii, hypos)
    # np.testing.assert_allclose(g.thetas, np.array([[0.0, 0.0]]))
    np.testing.assert_allclose(g.grid_pt_idx, np.array([0, 0]))
    np.testing.assert_allclose(g.is_regular, np.array([0, 0]))
    gp = prune(g)
    np.testing.assert_allclose(gp.grid_pt_idx, np.array([0]))
    np.testing.assert_allclose(gp.is_regular, np.array([0]))


def test_prune_no_surfaces():
    thetas = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    radii = np.full_like(thetas, 0.5)
    g = build_grid(thetas, radii, [])
    gp = prune(g)
    assert g == gp


def test_prune_twice_invariance(simple_grid):
    gp = prune(simple_grid)
    gpp = prune(gp)
    np.testing.assert_allclose(gp.thetas, gpp.thetas)
    np.testing.assert_allclose(gp.radii, gpp.radii)
    np.testing.assert_allclose(gp.vertices, gpp.vertices)
    np.testing.assert_allclose(gp.is_regular, gpp.is_regular)
    np.testing.assert_allclose(gp.null_truth, gpp.null_truth)
    np.testing.assert_allclose(gp.grid_pt_idx, gpp.grid_pt_idx)


n_arms = 3
n_theta_1d = 48


def py_grid():
    null_hypos = [HyperPlane(-np.identity(n_arms)[i], -2) for i in range(n_arms)]
    theta1d = [np.linspace(-3.5, 1.0, 2 * n_theta_1d + 1)[1::2] for i in range(n_arms)]
    theta = np.stack(np.meshgrid(*theta1d), axis=-1).reshape((-1, len(theta1d)))
    radii = np.empty(theta.shape)
    for i in range(theta.shape[1]):
        radii[:, i] = 0.5 * (theta1d[i][1] - theta1d[i][0])
    g = prune(build_grid(theta, radii, null_hypos))
    return g


def cpp_grid():
    import pyimprint.grid as grid

    # define null hypos
    null_hypos = []
    for i in range(n_arms):
        n = np.zeros(n_arms)
        # null is:
        # theta_i <= logit(0.1)
        # the normal should point towards the negative direction. but that also
        # means we need to negate the logit(0.1) offset
        n[i] = -1
        null_hypos.append(grid.HyperPlane(n, -2))

    gr = grid.make_cartesian_grid_range(
        n_theta_1d, np.full(n_arms, -3.5), np.full(n_arms, 1.0), 1
    )
    gr.create_tiles(null_hypos)
    gr.prune()
    return gr


def benchmark(f, iter=3):
    runtimes = []
    for i in range(iter):
        start = time.time()
        f()
        end = time.time()
        runtimes.append(end - start)
    return runtimes


if __name__ == "__main__":
    # Runtimes:
    # py_grid: [0.4465768337249756, 0.40291690826416016, 0.3762087821960449]
    # cpp_grid: [4.8662269115448, 4.6366658210754395, 4.647104024887085]6
    print("py_grid:", benchmark(py_grid))
    print("cpp_grid:", benchmark(cpp_grid))
