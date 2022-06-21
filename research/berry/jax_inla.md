---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.10.5 ('kevlar')
    language: python
    name: python3
---

```python
import sys

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import matplotlib.pyplot as plt


from berrylib.constants import Y_I, Y_I2, N_I, N_I2

```

```python
import numpy as np
import berrylib.fast_inla
import importlib
from functools import partial

importlib.reload(berrylib.fast_inla)
from berrylib.fast_inla import FastINLA, jax_opt

# assert jax.config.read("jax_enable_x64")

fi = FastINLA(n_arms=2)
# sigma2 = jnp.array(1e5)
# y = jnp.array([0, 0, 30, 30])[jnp.newaxis]
# n = jnp.array([30, 30, 30, 30])[jnp.newaxis]
# # fi.jax_inference(y, n)


def jax_faster_inv(D, S):
    """Compute the inverse of a diagonal matrix D plus a shift S.

    This function uses "Sherman-Morrison" formula:
    https://en.wikipedia.org/wiki/Sherman–Morrison_formula
    """
    D_inverse = 1.0 / D
    multiplier = -S / (1 + S * D_inverse.sum())
    M = multiplier * jnp.outer(D_inverse, D_inverse)
    M = M + jnp.diag(D_inverse)
    return M


# @jax.jit
def jax_opt(y, neg_precQ, fast_loop=True):
    # y = jnp.array([29.65146991, 27.51985572])
    n = jnp.array([35, 35])
    # neg_precQ = jnp.array([[-28764.0366170816, 28764.031617082037], [28764.031617082037, -28764.0366170816]])
    logit_p1 = -0.8472978603872036
    mu_0 = -1.34
    tol = 0.001

    def step(args):
        i, theta_max, hess_inv, go = args
        theta_m0 = theta_max - mu_0
        exp_theta_adj = jnp.exp(theta_max + logit_p1)
        nCeta = jnp.log(n) - jnp.log1p(exp_theta_adj) + (theta_max + logit_p1)
        diag = jnp.exp(nCeta - jnp.log1p(exp_theta_adj))
        nCeta = jnp.exp(nCeta)

        grad = neg_precQ @ theta_m0 + y - nCeta

        shift = neg_precQ[..., 0, 1]
        prec_d = jnp.diag(neg_precQ) - shift
        diag = prec_d - diag

        # Apply the regularization suggested in: https://arxiv.org/abs/2112.02089
        H = 10
        reg = jnp.sqrt(H * jnp.linalg.norm(grad))
        diag += reg

        hess_inv = jax_faster_inv(diag, shift)
        step = -hess_inv.dot(grad)

        probit_step = 1 / (1 + jnp.exp(-theta_max))
        probit_step = probit_step * (1 - probit_step) * step
        go = jnp.max(jnp.abs(probit_step)) > tol
        # go = jnp.sum(step**2) > tol**2
        return i + 1, theta_max + step, hess_inv, go

    n_arms = y.shape[0]
    theta_max0 = jnp.zeros(n_arms)
    init_args = (0, theta_max0, jnp.zeros((n_arms, n_arms)),  True)
    max_iter = 1000

    if fast_loop:
        out = jax.lax.while_loop( lambda args: ((args[0] < max_iter) & args[-1]), step, init_args)
    else:
        args = init_args
        step = jax.jit(step)
        converged = False
        for i in range(max_iter):
            args = step(args)
            out = args
            if not args[-1]:
                converged = True
                break
    i, theta_max, hess_inv, go = out
    return theta_max, hess_inv


# def run(y, i):
#     # cov = jnp.full((fi.n_arms, fi.n_arms), fi.mu_sig_sq)
#     # cov = cov + jnp.diag(jnp.full(fi.n_arms, sigma2))
#     # # shift, prec_d = berrylib.fast_inla.jax_faster_inv(jnp.repeat(sigma2, arms), 100)
#     # # neg_precQ = -(jnp.diag(prec_d) + shift)
#     # neg_precQ = jnp.linalg.inv(cov)
#     n = np.array([[35, 35]])

#     with open("out.txt", "a") as f:
#         def p(x):
#             f.write(str(x) + "\n")
#         p(y)
#         f.flush()
#     opt = jax_opt
#     theta_max, hess_inv = opt(
#     y[0],
#     n[0],
#     # fi.cov,
#     fi.neg_precQ[i],
#     # None,
#     fi.logit_p1,
#     fi.mu_0,
#     fi.tol,
#     )
#     if np.isnan(theta_max).any():
#         assert False, (y)
#     return theta_max

trials = 100_000

acc = 0
def test(y, j, fn=jax_opt):
    global acc
    if not fn(y, fi.neg_precQ[j]):
        acc +=1 #(i, y, j)

# y = jnp.array([ 7.28087852, 19.92078862])
# j = 0
# test(y, j, fn=partial(jax_opt, fast_loop=False))

fn = jax.jit(jax_opt)
acc = 0
for i in range(trials):
    arms = 2
    y = np.random.uniform(0, 35, size=(arms))
    if i % (trials // 10) == 0:
        print(i)
    for j in range(15):
        test(y, j, fn=fn)
# # %timeit run()
# _, exc, _, _ = fi.jax_inference(Y_I2, N_I2)
# exc[-1]
# ret = fi.jax_inference(Y_I2, N_I2)
# sigma2_post, exceedances, theta_max, theta_sigma=ret
# for r in ret:
# print(r.dtype)
# theta_max = jnp.array([0.1, 0.1, 0.1, 0.1])
# theta_m0 = theta_max - fi.mu_0
# theta_adj = theta_max + fi.logit_p1
# exp_theta_adj = jnp.exp(theta_adj)
# y = Y_I2
# n = N_I2

# jax_opt()
acc

```

```python
with open("/home/const/kevlar/out.txt") as f:
    arr = np.array(eval(f.read()))
arr.shape
np.linalg.slogdet(arr)
```

```python
# with jit:
x = np.array([2.3373992443084717, 2.337362289428711])
y = np.array([2.3381409645080566, 2.338103771209717])

def 

# without jit:
# [2.3373985290527344, 2.3373618125915527]
print(np.linalg.norm(x - y))
print(np.abs(x-y))

```

```python
def logit(x):
    return jnp.log(x) - jnp.log(1 - x)


def get_log_berry_likelihood(y, n):
    def log_berry_likelihood(theta, sigma_sq):
        ll = 0.0
        ll += dist.InverseGamma(0.0005, 0.000005).log_prob(sigma_sq)
        cov = jnp.full((4, 4), 100) + jnp.diag(jnp.repeat(sigma_sq, 4))
        ll += (
            dist.MultivariateNormal(-1.34, covariance_matrix=cov).log_prob(theta).sum()
        )
        ll += dist.BinomialLogits(logit(0.3) + theta, total_count=n).log_prob(y).sum()
        return ll

    return log_berry_likelihood


ll = get_log_berry_likelihood(y, n)
theta = jnp.array([0.0, 0, 0, 0])
sigma_sq = 1e1
ll(theta, sigma_sq)
grad = jax.grad(ll, 0)
hess = jax.jacobian(grad)
h = hess(theta, sigma_sq)
print(h)
print(jnp.linalg.inv(h))

```

```python
def mcmc_berry_model(y, n):
    # y, n = data
    mu = numpyro.sample("mu", dist.Normal(-1.34, 10))
    sigma2 = numpyro.sample("sigma2", dist.InverseGamma(0.0005, 0.000005))
    with numpyro.plate("j", 4):
        theta = numpyro.sample(
            "theta",
            dist.Normal(mu, jax.numpy.sqrt(sigma2)),
        )
        numpyro.sample(
            "y",
            dist.BinomialLogits(theta + 0, total_count=n),
            obs=y,
        )

```

# Stochastic Variational Inference

```python
import numpyro
from functools import partial
from numpyro.infer import Predictive, SVI, Trace_ELBO, init_to_uniform
from numpyro.contrib.einstein import RBFKernel, SteinVI
from numpyro.infer.autoguide import (
    AutoLaplaceApproximation,
    AutoDelta,
    AutoMultivariateNormal,
    AutoBNAFNormal,
)

data = Y_I2[-1], N_I2[-1]
y, n = data
model = mcmc_berry_model
optimizer = numpyro.optim.Adam(step_size=0.005)
# guide = AutoLaplaceApproximation(model)
guide = AutoBNAFNormal(model)
# guide = AutoMultivariateNormal(model)
# guide = AutoDelta(model, init_loc_fn=partial(init_to_uniform, radius=0.1))
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
svi_result = svi.run(jax.random.PRNGKey(0), 3_000, y, n)
params = svi_result.params
predictive = Predictive(guide, params=params, num_samples=100000)
samples = predictive(jax.random.PRNGKey(1), data)

```

```python
plt.plot(svi_result.losses)

```

```python
plt.figure(figsize=(10, 5))
for i in range(4):
    s = samples["theta"][:, i]
    s = s[jnp.newaxis]
    plt.hist(s, bins=100, density=True, label=f"$\\theta_{i}$", alpha=0.5)
plt.legend()
None

```

$ P(x_i | x_{-i}, y, \theta) = P(x, y, \theta) / P(x_{-i}, \theta, y) $

```python

```

2) gaussian approx at x=mu(theta)

* p_thetaIy = log_berry_likelihood(vars) - p_x_I_theta_y  

* Note the last term depends on a fixed theta too

2) p_thetaIy = log_berry_likelihood(vars) - .5 * log(-H(f(x_0)))  

3) p_xiIy = sum(p_xiItheta_y + p_thetaIy for theta in thetas)

* times dTheta

* p_xiItheta_y is skew normal approximation at MAP


$ P(t|y) = P(x, t, y) / gaussian(P(x | t, y)) $

$ P(x_i|t,y) = P(x, t, y) / gaussian(P(x-i|t,y)) $

$ P(x_i | y) = \sum(P(xi|t,y) * P(t|y) \Delta t) $

$ P(x_i | y) = \sum(P(xi|t,y) * P(t|y) \Delta t) $

```python
y = Y_I2[-1]
n = N_I2[-1]
log_berry_likelihood = get_log_berry_likelihood(y, n)
grad = jax.grad(log_berry_likelihood, 0)
hess = jax.jacobian(grad)
# theta = jnp.zeros(4)
key = jax.random.PRNGKey(0)
theta = jax.random.uniform(key, [4])
mu = jnp.zeros(1)
sigma = jnp.array(0.01)[jnp.newaxis]
vars = jnp.concatenate([theta, mu, sigma])
print(vars.shape)
jnp.exp(log_berry_likelihood(vars))
grad(vars)
h = hess(vars)
h

```

```python
N_ARMS = 4


def pack_vars(theta, mu, sigma):
    return jnp.concatenate([theta, mu, sigma])


def unpack_vars(vars):
    return vars[:N_ARMS], vars[N_ARMS], vars[N_ARMS + 1]


# print(hs.shape)
@jax.jit
def chol():
    hs = jnp.stack([h for i in range(4 * 16)])
    return jax.lax.linalg.cholesky(
        hs,
    )
    # return jnp.linalg.inv(hs)


# %timeit chol()

# @jax.jit
def optimize(theta, sigma, mask):
    log_berry_likelihood = get_log_berry_likelihood(y, n)
    grad = jax.grad(log_berry_likelihood, 0)
    hess = jax.jacobian(grad)
    # theta = jnp.zeros(4)
    mu = jnp.zeros(1)
    sigma = jnp.array(sigma)[jnp.newaxis]
    vars = pack_vars(theta, mu, sigma)
    # Do a newton iteration
    pvars = None
    for _ in range(10):
        g = grad(vars)
        h = hess(vars)
        g = g[mask]
        h = h[mask, mask]
        pvars = vars
        print(jnp.diag(h))
        update = jnp.linalg.solve(h, g)
        vars = vars.at[mask].add(-update)
        # assert not jnp.isnan(vars).any()
        # print(jnp.linalg.norm(pvars - vars))
    return vars

```

```python
theta = jnp.zeros(4)
sigma = 1e-8
mask = jnp.s_[0:5]
optimize(theta, sigma, mask)

```

```python

```

```python
sigmas = jnp.power(10, jnp.linspace(-8, 3, 10))
thetas = jnp.linspace(-10, 10, 10)
mask = jnp.s_[1:5]
varss = []
for sigma in sigmas:
    for theta in thetas:
        theta = jnp.zeros(4).at[0].set(theta)
        vars = optimize(theta, sigma, mask)
        varss.append(vars)
varss = jnp.stack(varss)

```

```python
varss.shape

```

```python
sigmas = jnp.log10(varss[:, N_ARMS + 1])
thetas = varss[:, :N_ARMS]
plt.scatter(sigmas, jnp.mean(thetas, 1))
plt.scatter(sigmas, jnp.std(thetas, 1))

```

```python

```

```python
x = 90
plt.scatter(thetas[x : x + 10, 0], thetas[x : x + 10, 1])

```

```python
thetas[x : x + 10]

```

```python
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def mcmc_berry_model(y, n):
    mu = numpyro.sample("mu", dist.Normal(-1.34, 10))
    sigma2 = numpyro.sample("sigma2", dist.InverseGamma(0.0005, 0.000005))
    with numpyro.plate("j", 2):
        theta = numpyro.sample(
            "theta",
            dist.Normal(mu, jax.numpy.sqrt(sigma2)),
        )
        numpyro.sample(
            "y",
            dist.BinomialLogits(theta + (np.log(0.3) - np.log(1 - 0.3)), total_count=n),
            obs=y,
        )


def do_mcmc(rng_key, y, n):
    nuts_kernel = NUTS(mcmc_berry_model)
    mcmc = MCMC(
        nuts_kernel,
        progress_bar=False,
        num_warmup=10_000,
        num_samples=1_000_000,
        thinning=10,
    )
    mcmc.run(rng_key, y, n)
    return mcmc.get_samples(group_by_chain=True)


s = do_mcmc(jax.random.PRNGKey(0), jnp.array([4, 8]), jnp.array([35, 35]))

```

```python
sig = s["sigma2"]
# sig = sig[np.newaxis]
sig = sig[:, :][0]
print(np.quantile(sig, [0.5]))
sig = np.log10(sig)
print(sig.shape)
fig = plt.figure(figsize=(7, 5))
plt.hist(sig, bins=10)
plt.xlabel("log10(sigma^2)")
fig.patch.set_alpha(1)

```

```python
def jax_calc_posterior_and_exceedances(
    theta_max,
    y,
    n,
    log_prior,
    neg_precQ,
    logprecQdet,
    hess_inv,
    sigma2_wts,
    logit_p1,
    mu_0,
    thresh_theta,
):
    theta_m0 = theta_max - mu_0
    theta_adj = theta_max + logit_p1
    exp_theta_adj = jnp.exp(theta_adj)
    logjoint = (
        0.5 * jnp.einsum("...i,...ij,...j", theta_m0, neg_precQ, theta_m0)
        + logprecQdet
        + jnp.sum(
            theta_adj * y[:, None] - n[:, None] * jnp.log(exp_theta_adj + 1),
            axis=-1,
        )
        + log_prior
    )

    log_sigma2_post = logjoint + 0.5 * jnp.log(jnp.linalg.det(-hess_inv))
    sigma2_post = jnp.exp(log_sigma2_post)
    sigma2_post /= jnp.sum(sigma2_post * sigma2_wts, axis=1)[:, None]

    theta_sigma = jnp.sqrt(jnp.diagonal(-hess_inv, axis1=2, axis2=3))
    exc_sigma2 = 1.0 - jax.scipy.stats.norm.cdf(
        thresh_theta,
        theta_max,
        theta_sigma,
    )
    exceedances = jnp.sum(
        exc_sigma2 * sigma2_post[:, :, None] * sigma2_wts[None, :, None], axis=1
    )
    return sigma2_post, exceedances

```
