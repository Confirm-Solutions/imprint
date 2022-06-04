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
