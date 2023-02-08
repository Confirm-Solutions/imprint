import copy
import time

import numpy as np
import pytest

import imprint.grid as grid
import imprint.planar_null as planar_null
from imprint.planar_null import HyperPlane
from imprint.planar_null import hypo

# NOTE: For developing tests, plotting a 2D grid is very useful:
# import matplotlib.pyplot as plt
# grid.plot_grid(g)
# plt.show()


def normalize(n):
    return n / np.linalg.norm(n)


def test_hypo():
    assert hypo("x < 0") == HyperPlane([-1], 0)
    assert hypo("x <= 0") == HyperPlane([-1], 0)
    assert hypo("x > 0") == HyperPlane([1], 0)
    assert hypo("x >= 0") == HyperPlane([1], 0)

    isq2 = 1.0 / np.sqrt(2)
    assert hypo("x < 1") == HyperPlane([-1], -1)
    assert hypo("x >= y") == HyperPlane([isq2, -isq2], 0)
    assert hypo("x + y < 0") == HyperPlane([-isq2, -isq2], 0)
    assert hypo("x + y < 1") == HyperPlane([-isq2, -isq2], -isq2)

    assert hypo("theta0 < 0") == HyperPlane([-1], 0)
    assert hypo("x0 < 0") == HyperPlane([-1], 0)

    assert hypo("y < 1") == HyperPlane([0, -1], -1)
    assert hypo("z < 1") == HyperPlane([0, 0, -1], -1)
    assert hypo("z < 0.2") == HyperPlane([0, 0, -1], -0.2)

    assert hypo("2*x < 0.2") == HyperPlane([-1], -0.1)
    assert hypo("2.1*x < 0.2") == HyperPlane([-1], -0.2 / 2.1)


def test_split2d():
    g = grid._raw_init_grid(
        np.array([[1.0, 1.0]]),
        np.array([[1.1, 1.1]]),
        0,
    )
    vertex_dist = np.array([[0.2, 0.2, -1.9, -1.9]])
    g = HyperPlane(np.array([-1, 0]), -0.1).split(g, vertex_dist)
    np.testing.assert_allclose(g.get_theta(), [[-0.0, 1.0], [1.1, 1.0]], atol=1e-6)
    np.testing.assert_allclose(g.get_radii(), [[0.1, 1.1], [1.0, 1.1]], atol=1e-6)


@pytest.fixture
def simple_grid():
    thetas = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    radii = np.full_like(thetas, 0.5)
    hypos = [HyperPlane(-np.identity(2)[i], -0.1) for i in range(2)]
    return grid._raw_init_grid(thetas, radii, 1).add_null_hypos(hypos)


n_bits, host_bits = grid._gen_short_uuids_one_batch.config
t_bits = 64 - n_bits - host_bits


def test_short_uuids():
    U = grid._gen_short_uuids(10, 1)
    assert np.unique(U).shape[0] == 10

    U2 = grid._gen_short_uuids(10, 1)
    assert U.dtype == np.uint64
    assert np.unique(U).shape[0] == 10
    assert U2[0] - U[0] == 2 ** (n_bits + host_bits)


def test_no_duplicate_uuids():
    n = int(2 ** (n_bits + 0.5))
    U = grid._gen_short_uuids(n, 1)
    assert np.unique(U).shape[0] == n

    n = 1000
    U = grid._gen_short_uuids(n, 1)
    U2 = grid._gen_short_uuids(n, 1)
    assert np.unique(np.concatenate((U, U2))).shape[0] == 2 * n


def test_lots_of_short_uuids():
    n = 2**n_bits
    uuids = grid._gen_short_uuids(n, 1)
    assert uuids[-1] - uuids[0] == 2 ** (n_bits + host_bits)
    assert np.unique(uuids).shape[0] == n


def test_add_null_hypos(simple_grid):
    g_active = simple_grid.prune_inactive()
    assert len(g_active.null_hypos) == 2
    np.testing.assert_allclose(
        np.concatenate((g_active.get_theta(), g_active.get_radii()), axis=1),
        np.array(
            [
                [-0.5, -0.5, 0.5, 0.5],
                [0.05, -0.5, 0.05, 0.5],
                [0.55, -0.5, 0.45, 0.5],
                [-0.5, 0.05, 0.5, 0.05],
                [-0.5, 0.55, 0.5, 0.45],
                [0.05, 0.05, 0.05, 0.05],
                [0.05, 0.55, 0.05, 0.45],
                [0.55, 0.05, 0.45, 0.05],
                [0.55, 0.55, 0.45, 0.45],
            ]
        ),
    )
    assert np.all(
        g_active.get_null_truth()
        == np.array(
            [[1, 1], [1, 1], [0, 1], [1, 1], [1, 0], [1, 1], [1, 0], [0, 1], [0, 0]]
        )
    )
    parent = g_active.df["parent_id"]
    assert parent.dtype == np.uint64
    assert ((parent == 0) | (parent.isin(simple_grid.df["id"]))).all()


def test_one_point_grid():
    g = grid._raw_init_grid(
        *grid._cartesian_gridpts(np.array([0]), np.array([1]), np.array([1])),
        worker_id=1
    )
    np.testing.assert_allclose(g.get_theta(), np.array([[0.5]]))
    np.testing.assert_allclose(g.get_radii(), np.array([[0.5]]))


def test_split_angled():
    Hs = [HyperPlane([2, -1], 0)]
    in_theta, in_radii = grid._cartesian_gridpts(
        np.full(2, -1), np.full(2, 1), np.full(4, 4)
    )
    g = (
        grid._raw_init_grid(in_theta, in_radii, worker_id=1)
        .add_null_hypos(Hs)
        .prune_alternative()
    )
    assert g.prune_inactive().n_tiles == 10
    np.testing.assert_allclose(g.get_radii()[-1], [0.125, 0.25])


def test_immutability():
    Hs = [HyperPlane([2, -1], 0)]
    in_theta, in_radii = grid._cartesian_gridpts(
        np.full(2, -1), np.full(2, 1), np.full(4, 4)
    )
    g = grid._raw_init_grid(in_theta, in_radii, worker_id=1)
    g_copy = copy.deepcopy(g)
    _ = g.add_null_hypos(Hs).prune_alternative()
    assert (g.df == g_copy.df).all().all()


def test_prune(simple_grid):
    gp = simple_grid.prune_alternative().prune_inactive()
    assert np.all(
        gp.get_null_truth()
        == np.array([[[1, 1], [1, 1], [0, 1], [1, 1], [1, 0], [1, 1], [1, 0], [0, 1]]])
    )


def check_index(g):
    assert np.all(g.df.index.values == np.arange(g.n_tiles))


def test_simple_indices(simple_grid):
    # All operations should leave the dataframe with a pandas index equal to
    # np.arange(n_tiles)
    g = grid.cartesian_grid([-1, -1], [1, 1], n=[2, 2])
    check_index(g)

    check_index(simple_grid)
    gp = simple_grid.prune_alternative()
    check_index(gp)
    gc = gp.concat(g)
    check_index(gc)


def test_column_inheritance():
    # All operations should leave the dataframe with a pandas index equal to
    # np.arange(n_tiles)
    g = grid.cartesian_grid([-1, -1], [1, 1], n=[2, 2])
    g.df["birthday"] = 1

    gs = g.add_null_hypos([planar_null.hypo("x < 0.1")], ["birthday"])
    assert (gs.df["birthday"] == 1).all()
    gp = gs.prune_alternative()
    assert (gp.df["birthday"] == 1).all()
    gc = gp.concat(g)
    assert (gc.df["birthday"] == 1).all()


def test_prune_no_surfaces():
    thetas = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    radii = np.full_like(thetas, 0.5)
    g = grid._raw_init_grid(thetas, radii, 1)
    gp = g.prune_alternative()
    assert g == gp


def test_prune_twice_invariance(simple_grid):
    gp = simple_grid.prune_alternative()
    gpp = gp.prune_alternative()
    np.testing.assert_allclose(gp.get_theta(), gpp.get_theta())
    np.testing.assert_allclose(gp.get_radii(), gpp.get_radii())
    np.testing.assert_allclose(gp.get_null_truth(), gpp.get_null_truth())


def test_refine():
    n_arms = 2
    theta, radii = grid._cartesian_gridpts(
        np.full(n_arms, -3.0), np.full(n_arms, 1.0), np.full(n_arms, 4)
    )

    null_hypos = [HyperPlane(-np.identity(n_arms)[i], 1.1) for i in range(n_arms)]
    g = (
        grid._raw_init_grid(theta, radii, 1)
        .add_null_hypos(null_hypos)
        .prune_alternative()
    )
    refine_g = g.prune_inactive().subset(np.array([0, 3, 4, 5]))
    new_g = refine_g.refine()
    np.testing.assert_allclose(new_g.get_radii()[:12], 0.25)
    np.testing.assert_allclose(new_g.get_radii()[-4:, 0], 0.225)
    np.testing.assert_allclose(new_g.get_radii()[-4:, 1], 0.25)

    pts_to_refine = np.array([[-2.5, -2.5], [-2.5, -0.5], [-2.5, 0.5], [-1.55, -2.5]])
    radius = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.45, 0.5]])

    for i in range(2):
        for j in range(2):
            subset = new_g.get_theta()[(2 * i + j) :: 4]
            correct = pts_to_refine + np.array([2 * i - 1, 2 * j - 1]) * radius * 0.5
            np.testing.assert_allclose(subset, correct)


def test_custom_null():

    from scipy.special import expit

    def null_curve(theta):
        return (
            -0.4 * expit(theta[..., 0])
            - 0.6 * expit(theta[..., 1])
            + (0.5 * 0.4 + 0.7 * 0.6)
        )

    class CurveNull(grid.NullHypothesis):
        def dist(self, theta):
            return null_curve(theta)

    g = grid.cartesian_grid([0, 0], [2, 1], n=[10, 10], null_hypos=[CurveNull()])
    assert g.df["active"].all()
    assert g.n_tiles == 33


# BENCHMARK

n_arms = 4
n_theta_1d = 10


def bench_f():
    return grid.cartesian_grid(
        np.full(n_arms, -3.5),
        np.full(n_arms, 1.0),
        np.full(n_arms, n_theta_1d),
        null_hypos=[HyperPlane(-np.identity(n_arms)[i], 2) for i in range(n_arms)],
    )


def benchmark(f, iter=3):
    runtimes = []
    for i in range(iter):
        start = time.time()
        f()
        end = time.time()
        runtimes.append(end - start)
    return runtimes


if __name__ == "__main__":
    print(benchmark(bench_f, iter=3))
