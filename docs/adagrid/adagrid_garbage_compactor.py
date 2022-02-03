from gridpt import GridPt
from binomial import Binomial2Arm
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

def adagrid_internal(gridpt, grid_q, grid_final, model, alpha, N_max, slack=alpha*0.1):
    # store upper bound at current grid point
    model.upper_bd(gridpt)

    # get full upper bound
    ub = gridpt.create_upper()

    # already a good estimate for ub: no need to tune N and eps further
    if (ub < alpha-slack) or (gridpt.N >= N_max):
        grid_final.append(gridpt)
        return

    # Compute z-score if N changed to N*2^d, d = dimension of gridpt
    N = gridpt.N
    d = len(gridpt.pt)
    N_factor = 2**d
    N_new = np.min(N * N_factor, N_max)
    N_ratio = N/N_new
    sigma_dN = gridpt.sigma / np.sqrt(N_new)
    mu_dN = gridpt.delta_0() + gridpt.delta_1() \
        + (gridpt.delta_0_u() + gridpt.delta_1_u()) * np.sqrt(N_ratio) \
        + gridpt.delta_2_u()
    z_dN = (alpha - mu_dN) / sigma_dN

    # Compute z-score if eps changed to eps/2
    sigma_deps = gridpt.sigma
    mu_deps = gridpt.delta_0() + gridpt.delta_1() \
        + gridpt.delta_1_u() / 2. \
        + gridpt.delta_2_u() / 4.
    z_deps = (alpha - mu_deps) / sigma_deps

    # Compare z-scores: larger the z-score, the more likely UpperBound < alpha.

    # 1) increase N and push gridpt into grid_q
    if z_dN > z_deps:
        gridpt.N = N_new
        grid_q.append(gridpt)

    # 2) decrease eps by adding children gridpts into grid_q
    else:
        bits = np.zeros(d)
        new_rad = gridpt.radius / 2
        for _ in range(2**d):
            new_pt = gridpt.pt + new_rad * (2*bits-1)

            # only add the new point if it's "viable".
            # The only check right now is if it's in the null
            if model.is_viable(new_pt):
                grid_q.put(GridPt(new_pt, new_rad, gridpt))

            # add 1 to bits
            for j in range(d-1, -1, -1):
                carry = (bits[j]+1)//2
                bits[j] = (bits[j] + 1) % 2
                if carry == 0:
                    break

def adagrid(lower, upper, model, alpha, init_size, max_iter, N_init, N_max):
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
    rnge = upper-lower
    radius = rnge / (2*init_size)
    theta_grids = (
        np.arange(lower[i]+radius[i], upper[i], step=2*radius[i])
        for i in range(len(lower))
    )

    # make full grid
    coords = np.meshgrid(*theta_grids)
    grid = np.concatenate(
        [c.flatten().reshape(-1,1) for c in coords],
        axis=1)

    # create initial queue of potential gridpts to look into further
    grid_q = queue.Queue()
    for pt in grid:
        if model.is_viable(pt):
            grid_q.put(GridPt(pt, radius, root_pt))

    # initialize lambdas
    grid_q_size = grid_q.qsize()
    for _ in range(grid_q_size):
        gridpt = grid_q.get()

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
        ax.plot_trisurf(pts[:,0], pts[:,1], z)
        plt.show()

    # If max iteration was reached, just output the current leaves
    # with the deepest nodes that we were supposed to prune further.
    if (itr == max_iter):
        grid_final = list(set().union(grid_final, grid_q.queue))

    return np.array(grid_final)

if __name__ == '__main__':
    model = Binomial2Arm()
    grid = adagrid(lower=np.array([-0.02, -0.02]),
                   upper=np.array([0.02, 0.02]),
                   model=model,
                   alpha=0.025,
                   init_size=8,
                   max_iter=8,
                   N_init=1000,
                   N_max=64000)
    grid_raw = np.array([pt.pt for pt in grid])
    N_raw = np.array([pt.N for pt in grid])
    plt.scatter(grid_raw[:,0], grid_raw[:,1], s=1, alpha=N_raw/np.max(N_raw))
    plt.show()
