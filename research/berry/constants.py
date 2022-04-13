import numpy as np

# I got this data by deconstructing the graphs in in Figure 1 of Berry et al 2013.
N_I = np.array([[i] * 4 for i in [10, 15, 20, 25, 30, 35]])

Y_I = np.array(
    [
        [1, 6, 3, 3],
        [3, 8, 5, 4],
        [6, 9, 7, 5],
        [7, 10, 8, 7],
        [8, 10, 9, 8],
        [11, 11, 10, 9],
    ]
)
DATA = np.stack((Y_I, N_I), axis=2)

N_I2 = np.array(
    [
        [10, 10, 10, 10],
        [15, 15, 15, 15],
        [20, 20, 20, 20],
        [20, 20, 25, 25],
        [20, 20, 30, 30],
        [20, 20, 35, 35],
    ]
)

Y_I2 = np.array(
    [
        [0, 1, 3, 3],
        [0, 1, 4, 5],
        [0, 1, 6, 6],
        [0, 1, 6, 7],
        [0, 1, 7, 8],
        [0, 1, 9, 10],
    ],
    dtype=np.float64,
)

DATA2 = np.stack((Y_I2, N_I2), axis=2)
