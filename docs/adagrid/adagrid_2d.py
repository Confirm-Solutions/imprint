# flake8: noqa
# The above line should be removed, but currently the code in this file is
# incorrect in several ways and it is unclear to me whether this is dead code or
# simply temporarily broken.
import queue

import matplotlib.pyplot as plt
import numpy as np
from binomial import Binomial2Arm
from gridpt import GridPt
from scipy.stats import norm


def adagrid_internal(gridpt, grid_q, grid_final, model, alpha, N_max):
    # store upper bound at current grid point
    model.upper_bd(gridpt)

    # get full upper bound
    ub = gridpt.create_upper()

    itr = 0
    while ub >= alpha and itr < 20:
        model.tune_gridpt(gridpt)
        ub = gridpt.create_upper()
        ++itr

    # delta_0' + delta_1' ~ N(delta_0 + delta_1, sigma^2)
    # sigma = sd([delta_0'_i + v^* delta_1'_i])
    # P(Ub > alpha) ~~ 1-NCDF((ub-alpha)/sigma) decrease
    while ((ub - alpha) / sigma) < norm.isf(0.05):
        model.tune_gridpt(gridpt)
        ub = gridpt.create_upper()
        # update sigma

    print(
        "{p}:\n\tub={ub}\n\tub_old={ub_old}".format(p=gridpt.pt, ub=ub, ub_old=ub_old)
    )

    # if we have to shrink grid
    if shrink_grid:
        d = len(gridpt.pt)
        bits = np.zeros(d)
        new_rad = gridpt.radius / 2
        for _ in range(2**d):
            new_pt = gridpt.pt + new_rad * (2 * bits - 1)

            # only add the new point if it's "viable".
            # The only check right now is if it's in the null
            if model.is_viable(new_pt):
                grid_q.put(GridPt(new_pt, new_rad, gridpt))

            # add 1 to bits
            for j in range(d - 1, -1, -1):
                carry = (bits[j] + 1) // 2
                bits[j] = (bits[j] + 1) % 2
                if carry == 0:
                    break
    else:
        grid_final.append(gridpt)


def adagrid(
    lower, upper, model, alpha=0.025, init_size=2, N_init=1000, N_max=100000, max_iter=2
):
    # set-up root node for special behavior
    root_pt = GridPt(None, None, None)
    root_pt.N = N_init
    root_pt.delta_0 = np.Inf
    root_pt.delta_0_u = np.Inf
    root_pt.delta_1 = np.Inf
    root_pt.delta_1_u = np.Inf
    root_pt.delta_2_u = np.Inf
    root_pt.delta_0_ci_lower = np.Inf
    root_pt.delta_0_ci_upper = np.Inf

    # make initial 1d grid
    rnge = upper - lower
    radius = rnge / (2 * init_size)
    theta_grids = (
        np.arange(lower[i] + radius[i], upper[i], step=2 * radius[i])
        for i in range(len(lower))
    )

    # make full grid
    coords = np.meshgrid(*theta_grids)
    grid = np.concatenate([c.flatten().reshape(-1, 1) for c in coords], axis=1)

    grid_q = queue.Queue()
    for pt in grid:
        if model.is_viable(pt):
            grid_q.put(GridPt(pt, radius, root_pt))

    # Final list of nodes to actually compute upper bound for.
    # Essentially the leaves of the tree we are building.
    grid_final = list()

    itr = 0
    while (not grid_q.empty()) and (itr < max_iter):
        # TODO: all grid_plt related stuff is temporary
        grid_plt = []

        # run through current queue and update the queue
        grid_q_size = grid_q.qsize()
        for _ in range(grid_q_size):
            gridpt = grid_q.get()
            adagrid_internal(gridpt, grid_q, grid_final, model, alpha, N_max)
            # TODO
            grid_plt.append(gridpt)
        itr += 1

        # TODO: temporary code here
        # plot the upper bound for each of the points
        grid_plt = list(set().union(grid_final, grid_plt))
        pts = np.array([pt.pt for pt in grid_plt])
        z = np.array([gp.create_upper() for gp in grid_plt])
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_trisurf(pts[:, 0], pts[:, 1], z)
        plt.show()

    # If max iteration was reached, just output the current leaves
    # with the deepest nodes that we were supposed to prune further.
    if itr == max_iter:
        grid_final = list(set().union(grid_final, grid_q.queue))

    return np.array(grid_final)


if __name__ == "__main__":
    model = Binomial2Arm()
    grid = adagrid(
        lower=np.array([-0.02, -0.02]),
        upper=np.array([0.02, 0.02]),
        model=model,
        alpha=0.025,
        init_size=16,
        max_iter=6,
        N_max=10000,
    )
    grid_raw = np.array([pt.pt for pt in grid])
    N_raw = np.array([pt.N for pt in grid])
    plt.scatter(grid_raw[:, 0], grid_raw[:, 1], s=1, alpha=N_raw / np.max(N_raw))
    plt.show()
