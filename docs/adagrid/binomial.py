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
            curr_grad = np.array([x_control, x_treat]) - self.n_sample*p
            gridpt.grad += curr_grad
            gridpt.kernel_trick += np.outer(curr_grad, curr_grad)

        ## finalize upper bound
        gridpt.delta_0 /= gridpt.N
        gridpt.grad /= gridpt.N

        gridpt.delta_0_u = \
            norm.isf(self.delta/2.) * \
            np.sqrt(gridpt.delta_0 * (1-gridpt.delta_0) / gridpt.N)

        v_star = np.abs(gridpt.grad)
        gridpt.delta_1 = np.dot(gridpt.radius, v_star)

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

        # 1/N sum_i (1_F_i + v^T df_hat_i)^2 - (mean)^2
        # = 1/N sum_i (1_F_i + 2v^T df_hat_i + v^T df_hat_i df_hat_i^T v) - (mean)^2
        # = delta_0 + 2*delta_1 + v^T (1/N sum_i df_hat_i df_hat_i^T) v - (mean)^2
        # mean = delta_0 + delta_1
        gridpt.sigma = np.sqrt(gridpt.delta_0 \
            + 2*gridpt.delta_1 \
            + v_star.dot(gridpt.kernel_trick.dot(v)) \
            - (gridpt.delta_0 + gridpt.delta_1)**2)
