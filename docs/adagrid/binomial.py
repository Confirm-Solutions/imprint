import numpy as np
from scipy.stats import norm


class Binomial2Arm:
    def __init__(self):
        self.n_sample = 250
        self.alpha_minus_target = None

        self.da_dthresh = None
        self.thresh = None
        self.thresh_minus = None

        self.null_hypo = lambda p: p[1] <= p[0]
        self.seed = 1324
        self.alpha_target = 0.025
        self.delta = 0.025

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def is_viable(self, theta):
        return self.null_hypo(Binomial2Arm.sigmoid(theta))

    def simulate_once(self, gridpt):
        # generate RNG
        unifs = np.random.uniform(size=(self.n_sample, 2))

        p = Binomial2Arm.sigmoid(gridpt.pt)

        # construct binomials
        x_control = np.sum(unifs[:, 0] < p[0])
        x_treat = np.sum(unifs[:, 1] < p[1])

        # construct z-stat
        p_control = x_control / self.n_sample
        p_treat = x_treat / self.n_sample
        var = (p_control * (1 - p_control) + p_treat * (1 - p_treat)) / self.n_sample
        z = p_treat - p_control
        if var <= 0:
            z = np.Inf * np.sign(z)
        else:
            z /= np.sqrt(var)

        return np.array([x_control, x_treat]), z

    def simulate(self, gridpt):
        # prepare some members of gridpt
        p = Binomial2Arm.sigmoid(gridpt.pt)
        if not self.null_hypo(p):
            return

        # set seed
        np.random.seed(self.seed)

        gridpt.grad = np.zeros(2)
        gridpt.grad_minus = np.zeros(2)

        # run through each simulation and update upper bound
        for i in range(gridpt.N):
            X, z = self.simulate_once(gridpt)

            ## accumulate upper bound quantities

            curr_grad = X - self.n_sample * p

            if z > self.thresh_minus:
                gridpt.delta_0_minus += 1
                gridpt.grad_minus += curr_grad

            # rejected if above thresh and is under null
            if z > self.thresh:
                gridpt.delta_0 += 1
                gridpt.grad += curr_grad
                gridpt.kernel_trick += np.outer(curr_grad, curr_grad)

        ## finalize upper bound
        gridpt.delta_0 /= gridpt.N
        gridpt.delta_0_minus /= gridpt.N
        gridpt.grad_minus /= gridpt.N
        gridpt.delta_1_minus = np.dot(gridpt.radius, np.abs(gridpt.grad_minus))
        gridpt.grad /= gridpt.N
        gridpt.kernel_trick /= gridpt.N

    def upper_bd(self, gridpt):
        # simulate
        self.simulate(gridpt)

        # mean parameter
        p = Binomial2Arm.sigmoid(gridpt.pt)

        gridpt.delta_0_u = norm.isf(self.delta / 2.0) * np.sqrt(
            gridpt.delta_0 * (1 - gridpt.delta_0) / gridpt.N
        )

        v_star = gridpt.radius * np.sign(gridpt.grad)
        gridpt.delta_1 = np.dot(gridpt.radius, np.abs(gridpt.grad))

        gridpt.delta_1_u = np.sqrt(
            np.dot(gridpt.radius**2, p * (1 - p))
            * self.n_sample
            * (2.0 / self.delta - 1.0)
            / gridpt.N
        )

        gridpt.delta_2_u = 0
        for k in range(len(p)):
            pk_lower = Binomial2Arm.sigmoid(gridpt.pt[k] - gridpt.radius[k])
            pk_upper = Binomial2Arm.sigmoid(gridpt.pt[k] + gridpt.radius[k])
            if pk_lower <= 0.5 and 0.5 <= pk_upper:
                gridpt.delta_2_u += 0.25 * gridpt.radius[k] ** 2
            else:
                lower = pk_lower - 0.5
                upper = pk_upper - 0.5
                max_at_upper = np.abs(upper) < np.abs(lower)
                max_endpt = pk_upper if max_at_upper else pk_lower
                gridpt.delta_2_u += max_endpt * (1 - max_endpt) * gridpt.radius[k] ** 2
        gridpt.delta_2_u *= self.n_sample / 2.0

        gridpt.delta_0_ci_lower = (
            gridpt.delta_0 - (2 * gridpt.delta_0 * (1 - gridpt.delta_0)) / gridpt.N
        )
        gridpt.delta_0_ci_upper = (
            gridpt.delta_0 + (2 * gridpt.delta_0 * (1 - gridpt.delta_0)) / gridpt.N
        )

        # 1/N sum_i (1_F_i + v^T df_hat_i)^2 - (mean)^2
        # = 1/N sum_i (1_F_i + 2v^T df_hat_i + v^T df_hat_i df_hat_i^T v) - (mean)^2
        # = delta_0 + 2*delta_1 + v^T (1/N sum_i df_hat_i df_hat_i^T) v - (mean)^2
        # mean = delta_0 + delta_1
        gridpt.sigma = np.sqrt(
            gridpt.delta_0
            + 2 * gridpt.delta_1
            + v_star.dot(gridpt.kernel_trick.dot(v_star))
            - (gridpt.delta_0 + gridpt.delta_1) ** 2
        )

    def initial_thresh(self, gridpt):
        p = Binomial2Arm.sigmoid(gridpt.pt)
        if not self.null_hypo(p):
            return

        # set seed
        np.random.seed(self.seed)

        z_vec = np.array([self.simulate_once(gridpt)[1] for _ in range(gridpt.N)])
        np.sort(z_vec)
        alpha = self.alpha_target
        self.alpha_minus_target = alpha - 2 * np.sqrt(alpha * (1 - alpha) / gridpt.N)
        thr = np.quantile(z_vec, 1 - alpha)
        thr_minus = np.quantile(z_vec, 1 - self.alpha_minus_target)

        return thr_minus, thr
