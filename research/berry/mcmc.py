import scipy.stats
import numpy as np

def proposal(x, sigma=0.25):
    rv = scipy.stats.norm.rvs(x, sigma, size=(x.shape[0], 6))

    while np.any(rv[:, 5] < 0):
        # Truncate normal distribution for the precision at 0.
        bad = rv[:, 5] < 0
        badx = x[bad]
        rv[bad, 5] = scipy.stats.norm.rvs(badx[:, 5], sigma, size=badx.shape[0])
    ratio = 1
    return rv, ratio


def mcmc(y, n, iterations=2000, burn_in=500, skip=2):
    def joint(xstar):
        a = xstar[:, -2]
        Qv = xstar[:, -1]
        return np.exp(calc_log_joint(xstar[:, :4], y, n, a, Qv))

    M = y.shape[0]
    x = np.zeros((M, 6))
    x[:, -1] = 1

    Jx = joint(x)
    x_chain = [x]
    J_chain = [Jx]
    accept = [np.ones(M)]

    for i in range(iterations):
        xstar, ratio = proposal(x)

        Jxstar = joint(xstar)
        hastings_ratio = (Jxstar * ratio) / Jx
        U = np.random.uniform(size=M)
        should_accept = U < hastings_ratio
        x[should_accept] = xstar[should_accept]
        Jx[should_accept] = Jxstar[should_accept]

        accept.append(should_accept)
        x_chain.append(x.copy())
        J_chain.append(Jx.copy())
    x_chain = np.array(x_chain)
    J_chain = np.array(J_chain)
    accept = np.array(accept).T

    x_chain_burnin = x_chain[burn_in::skip]

    ci025n = int(x_chain_burnin.shape[0] * 0.025)
    ci975n = int(x_chain_burnin.shape[0] * 0.975)
    results = dict(
        CI025=np.empty(x.shape), CI975=np.empty(x.shape), mean=np.empty(x.shape)
    )
    for j in range(6):
        x_sorted = np.sort(x_chain_burnin[:, :, j], axis=0)
        x_mean = x_sorted.mean(axis=0)
        results["CI025"][:, j] = x_sorted[ci025n]
        results["mean"][:, j] = x_mean
        results["CI975"][:, j] = x_sorted[ci975n]
    return results