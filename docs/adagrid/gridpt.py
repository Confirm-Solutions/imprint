import numpy as np


class GridPt:
    def __init__(self, pt, radius, parent):
        self.pt = pt
        self.parent = parent
        self.N = 0 if parent is None else parent.N
        self.radius = radius
        self.grad = None
        self.grad_minus = None
        self.delta_0 = 0
        self.delta_0_minus = 0
        self.delta_0_u = 0
        self.delta_1 = 0
        self.delta_1_minus = 0
        self.delta_1_u = 0
        self.delta_2_u = 0
        self.delta_0_ci_lower = 0
        self.delta_0_ci_upper = 0
        self.sigma = 0
        self.kernel_trick = (
            None if radius is None else np.zeros(shape=(len(radius), len(radius)))
        )

    def __repr__(self):
        return "{pt}, N={N}, deltas={deltas}, sigma={sigma}\n".format(
            pt=self.pt,
            N=self.N,
            deltas=[
                self.delta_0,
                self.delta_1,
                self.delta_0_u,
                self.delta_1_u,
                self.delta_2_u,
            ],
            sigma=self.sigma,
        )

    def create_upper(self):
        return (
            self.delta_0
            + self.delta_0_u
            + self.delta_1
            + self.delta_1_u
            + self.delta_2_u
        )
