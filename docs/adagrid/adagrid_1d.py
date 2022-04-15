import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def f(x):
    return np.abs(
        np.cos(x) * 3 - np.sin(2 * x) - x + x**2 - 1 / 3.0 * x**3 * np.cos(x)
    )


def adagrid_1d(lower, upper, f, sigma=1, n_batch=10, n_max=100, seed=0, tol=1e-2):
    # set seed
    np.random.seed(seed)

    # create initial values
    x = np.linspace(lower, upper, n_batch)
    fx = f(x)
    # sd = np.maximum(np.abs(fx), 1e-8)
    sd = sigma * np.ones(len(x))
    w_sum = np.sum(fx)
    w = fx / w_sum

    # compute loss-function
    p = np.array([np.dot(w, norm.pdf(xi, loc=x, scale=sd)) for xi in x])
    L = np.dot((p - w) ** 2, w)
    L_prev = np.Inf

    n_pts = n_batch

    while (n_pts < n_max) and (np.abs(L - L_prev) > L * tol):
        z_new = np.random.choice(len(w), size=n_batch, replace=True, p=w)
        mean_new = x[z_new]
        sd_new = sd[z_new]
        x_new = np.random.normal(loc=mean_new, scale=sd_new)

        for i in range(len(x_new)):
            while (x_new[i] < lower) or (x_new[i] > upper):
                x_new[i] = np.random.normal(mean_new[i], sd_new[i])

        fx_new = f(x_new)
        x = np.append(x, x_new)
        # sd = np.append(sd, 1./np.maximum(np.abs(fx_new), 1e-8))
        sd = sigma * np.ones(len(x))
        w_sum_new = np.sum(fx_new)
        w *= w_sum / (w_sum + w_sum_new)
        w_sum += w_sum_new
        w = np.append(w, fx_new / w_sum)
        n_pts += n_batch

        # compute loss-function
        p = np.array([np.dot(w, norm.pdf(xi, loc=x, scale=sd)) for xi in x])
        L_prev = L
        L = np.dot((p - w) ** 2, w)

    return x


if __name__ == "__main__":
    x = adagrid_1d(-10, 10, f, sigma=1, n_batch=10, n_max=1000, tol=1e-2)
    print("len(x) = {n}".format(n=len(x)))
    x_ord = np.argsort(x)
    x_even = np.linspace(np.min(x), np.max(x), 1000)
    plt.plot(x[x_ord], f(x)[x_ord], ls="-", marker=".")
    plt.plot(x_even, f(x_even), ls="--")
    plt.show()
