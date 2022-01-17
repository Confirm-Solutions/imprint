import numpy as np
from scipy.stats import norm
from copy import copy
import queue
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class GridPt():
    def __init__(self, pt, radius, parent):
        self.pt = pt
        self.parent = parent
        self.N = 0 if parent is None else parent.N
        self.radius = radius
        self.grad = None
        self.delta_0 = 0
        self.delta_0_u = 0
        self.delta_1 = 0
        self.delta_1_u = 0
        self.delta_2_u = 0
        self.delta_0_ci_lower = 0
        self.delta_0_ci_upper = 0

    def __repr__(self):
        return "{pt}, N={N}\n".format(pt=self.pt, N=self.N)

    def create_upper(self):
        return self.delta_0 + \
                self.delta_0_u + \
                self.delta_1 + \
                self.delta_1_u + \
                self.delta_2_u

class Binomial2Arm():
    def __init__(self):
        self.n_sample = 250
        self.seed = 0
        self.thresh = 1.96 # TODO: somehow be generalizable
        self.null_hypo = lambda p: p[1] <= p[0]
        self.delta = 0.025

    @staticmethod
    def sigmoid(x):
        return 1./(1.+np.exp(-x))

    def is_viable(self, theta):
        return self.null_hypo(Binomial2Arm.sigmoid(theta))

    def upper_bd(self, gridpt):
        # prepare some members of gridpt
        gridpt.grad = np.zeros(2)
        p = Binomial2Arm.sigmoid(gridpt.pt)

        # set seed
        np.random.seed(self.seed)

        if not self.null_hypo(p):
            return

        # run through each simulation and update upper bound
        for i in range(gridpt.N):
            # generate RNG
            unifs = np.random.uniform(size=(self.n_sample, 2))

            # construct binomials
            x_control = np.sum(unifs[:,0] < p[0])
            x_treat = np.sum(unifs[:,1] < p[1])

            # construct z-stat
            p_control = x_control / self.n_sample
            p_treat = x_treat / self.n_sample
            var = (p_control*(1-p_control) + p_treat*(1-p_treat)) / self.n_sample
            z = p_treat - p_control
            if var <= 0:
                z = np.Inf * np.sign(z)
            else:
                z /= np.sqrt(var)

            ## accumulate upper bound quantities

            # rejected if above thresh and is under null
            rej = (z > self.thresh)

            if not rej:
                continue

            gridpt.delta_0 += 1
            gridpt.grad += np.array([x_control, x_treat]) - self.n_sample*p

        ## finalize upper bound
        gridpt.delta_0 /= gridpt.N
        gridpt.grad /= gridpt.N

        gridpt.delta_0_u = \
            norm.isf(self.delta/2.) * \
            np.sqrt(gridpt.delta_0 * (1-gridpt.delta_0) / gridpt.N)

        gridpt.delta_1 = np.dot(gridpt.radius, np.abs(gridpt.grad))

        gridpt.delta_1_u = np.sqrt(
            np.dot(gridpt.radius**2, p*(1-p)) *
            self.n_sample *
            (2./self.delta - 1.) / gridpt.N
        )

        gridpt.delta_2_u = 0
        for k in range(len(p)):
            pk_lower = Binomial2Arm.sigmoid(gridpt.pt[k]-gridpt.radius[k])
            pk_upper = Binomial2Arm.sigmoid(gridpt.pt[k]+gridpt.radius[k])
            if pk_lower <= 0.5 and 0.5 <= pk_upper:
                gridpt.delta_2_u += 0.25 * gridpt.radius[k]**2
            else:
                lower = pk_lower - 0.5
                upper = pk_upper - 0.5
                max_at_upper = np.abs(upper) < np.abs(lower)
                max_endpt = pk_upper if max_at_upper else pk_lower
                gridpt.delta_2_u += max_endpt * (1-max_endpt) * gridpt.radius[k]**2
        gridpt.delta_2_u *= self.n_sample / 2.

        gridpt.delta_0_ci_lower = \
            gridpt.delta_0 - (2*gridpt.delta_0*(1-gridpt.delta_0))/gridpt.N
        gridpt.delta_0_ci_upper = \
            gridpt.delta_0 + (2*gridpt.delta_0*(1-gridpt.delta_0))/gridpt.N

    def tune_gridpt(gridpt, N_max):
        delta_ub_eps = 0
        if np.maximum(gridpt.radius) > 1e-6:
            delta_ub_eps = 0.5 * (gridpt.delta_1 + gridpt.delta_1_u) + 0.75 * gridpt.delta_2_u

        delta_ub_N = 0
        if gridpt.N < N_max:
            gridpt_tmp = GridPt(gridpt.pt, gridpt.radius, gridpt.parent)
            model.seed += 1
            model.upper_bound(gridpt_tmp)
            model.seed -= 1
            ndelta_0 = 0.5 * (gridpt.delta_0 - gridpt_tmp.delta_0)
            ndelta_0_u = gridpt.delta_0_u - norm.isf(self.delta/2.) * \
                        np.sqrt(ndelta_0 * (1-ndelta_0) / (2*gridpt.N))
            ndelta_1 = 0.5 * (gridpt.delta_1 - gridpt_tmp.delta_1)
            ndelta_1_u = 0.5 * gridpt.delta_1_u
            delta_ub_N = ndelta_0 + ndelta_0_u + ndelta_1 + ndelta_1_u

        delta_ub_lmda = 0

        new_thresh = model.delta_lmda(gridpt)

        imax = np.argmax([delta_ub_eps, delta_ub_N, delta_ub_lmda])

        # maximum occurs for changing lambda
        if imax == 2:
            model.thresh = new_thresh
            break

def adagrid_internal(gridpt, grid_q, grid_final, model, alpha, N_max):
    # store upper bound at current grid point
    model.upper_bd(gridpt)

    # get full upper bound
    ub = gridpt.create_upper()

    while ub >= alpha:
        model.tune_gridpt(gridpt)
        ub = gridpt.create_upper()

    print("{p}:\n\tub={ub}\n\tub_old={ub_old}".format(p=gridpt.pt, ub=ub, ub_old=ub_old))

    # if we have to shrink grid
    if shrink_grid:
        d = len(gridpt.pt)
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
    else:
        grid_final.append(gridpt)

def adagrid(lower, upper, model, alpha=0.025,
            init_size=2, N_init=1000, N_max=100000, max_iter=2):
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
                   init_size=16,
                   max_iter=6,
                   N_max=10000)
    grid_raw = np.array([pt.pt for pt in grid])
    N_raw = np.array([pt.N for pt in grid])
    plt.scatter(grid_raw[:,0], grid_raw[:,1], s=1, alpha=N_raw/np.max(N_raw))
    plt.show()
