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
        self.sigma = 0
        self.kernel_trick = np.zeros(len(radius))

    def __repr__(self):
        return "{pt}, N={N}\n".format(pt=self.pt, N=self.N)

    def create_upper(self):
        return self.delta_0 + \
                self.delta_0_u + \
                self.delta_1 + \
                self.delta_1_u + \
                self.delta_2_u
