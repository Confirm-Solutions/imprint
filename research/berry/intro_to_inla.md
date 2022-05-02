---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.10.2 ('kevlar')
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from scipy.special import logit

import sys
sys.path.append('../../python/example/berry')
import util

n_i = np.array([20, 20, 35, 35])
y_i = np.array([0, 1, 9, 10], dtype=np.float64)
data = np.stack((y_i, n_i), axis=1)

mu_0 = -1.34
mu_sig_sq = 100.0
logit_p1 = logit(0.3)
```

## Section 1: compute $p(\sigma^2|y)$


## 1.1: Construct $\sigma^2$ quadrature rule

and compute values dependent only on $\sigma^2$:
* precision matrix for p(\theta | \sigma^2): inverse covariance.
* determinant of precision matrix
* values of the prior.

```python
sigN = 15
sigma2_rule = util.log_gauss_rule(sigN, 1e-6, 1e3)
arms = np.arange(4)
cov = np.full((sigN, 4, 4), mu_sig_sq)
cov[:, arms, arms] += sigma2_rule.pts[:, None]

precQ = np.linalg.inv(cov)
precQdet = np.linalg.det(precQ)
log_prior = scipy.stats.invgamma.logpdf(sigma2_rule.pts, 0.0005, scale=0.000005)
```

## 1.2: find the maximum of $p(\theta | y, \sigma^2)$

for each value of $\sigma^2$.

```python
tol = 1e-8
theta_max = np.zeros((sigN, 4))
for i in range(100):
    theta_m0 = theta_max - mu_0
    theta_adj = theta_max + logit_p1
    grad = (
        -np.sum(precQ * theta_m0[:, None, :], axis=-1)
        + y_i
        - (n_i * np.exp(theta_adj) / (np.exp(theta_adj) + 1))
    )
    hess = -precQ.copy()
    hess[:, arms, arms] -= n_i * np.exp(theta_adj) / ((np.exp(theta_adj) + 1) ** 2)
    step = -np.linalg.solve(hess, grad)
    theta_max += step

    if np.max(np.linalg.norm(step, axis=-1)) < tol:
        print(i)
        break
```

## 1.3 Laplace approximation for calculating $p(\theta|y)$

```python
theta_m0 = theta_max - mu_0
theta_adj = theta_max + logit_p1
logjoint = (
    -0.5 * np.einsum("...i,...ij,...j", theta_m0, precQ, theta_m0)
    + 0.5 * np.log(precQdet)
    + np.sum(theta_adj * y_i - n_i * np.log(np.exp(theta_adj) + 1), axis=-1)
    + log_prior
)

# The last step will be sufficiently small that we shouldn't need to update the
# hessian
# hess = -precQ.copy()
# hess[:, arms, arms] -= n_i * np.exp(theta_adj) / ((np.exp(theta_adj) + 1) ** 2)
log_sigma2_post = logjoint - 0.5 * np.log(np.linalg.det(-hess))
# This can be helpful for avoiding overflow.
# log_sigma2_post -= np.max(log_sigma2_post, axis=-1) - 600
sigma2_post = np.exp(log_sigma2_post)
sigma2_post /= np.sum(sigma2_post * sigma2_rule.wts)
plt.plot(np.log10(sigma2_rule.pts), sigma2_post * sigma2_rule.wts)
plt.show()
```

```python
sigma2_post
```

# Section 2: computing latent variable marginals: $p(\theta_i | y)$


```python
arm_idx = 0
ti_N = 61
ti_rule = util.simpson_rule(ti_N, -6.0, 2.0)
```


## 2.0: Do it with full numerical integration (slow)


```python
import fast_inla
import quadrature

fi = fast_inla.FastINLA(n_arms=4)
quad_p_ti_g_y = quadrature.integrate(
    fi, y_i[None, :], n_i[None, :], fixed_arm_dim=arm_idx, fixed_arm_values=ti_rule.pts,
    n_theta=21
)
quad_p_ti_g_y /= np.sum(quad_p_ti_g_y * ti_rule.wts, axis=1)[:, None]

```

```python
quad_p_ti_g_y
```

<!-- #region -->


## 2.1: Gaussian approximation of $p(\theta_i|y, \sigma^2)$

When we calculated the posterior of the hyperparameters above in section 1.3, we developed a multi-variate normal approximation for $p(\theta|y, \sigma^2)$. We can re-use this approximation now to compute $p(\theta_i | y)$. 

The final step is to evaluate at a range of $\theta_i$ and then integrate over $\sigma^2$.
<!-- #endregion -->

```python
theta_i_sigma = np.sqrt(np.diagonal(-np.linalg.inv(hess), axis1=1, axis2=2))
theta_i_mu = theta_max
gaussian_pdf = scipy.stats.norm.pdf(
    ti_rule.pts[:, None],
    theta_i_mu[None, :, arm_idx],
    theta_i_sigma[None, :, arm_idx],
)
gaussian_p_ti_g_y = np.sum(
    gaussian_pdf * sigma2_post * sigma2_rule.wts[None, :], axis=1
)

plt.plot(ti_rule.pts, gaussian_p_ti_g_y, "r-o", markersize=3, label="INLA-Gaussian")
plt.plot(ti_rule.pts, quad_p_ti_g_y[0], "b-o", markersize=3, label="Quad")
plt.legend()
plt.show()
```

## 2.1a: redo it for all the arms:

```python
for arm_idx in range(4)[::-1]:
    print(arm_idx)
    quad_p_ti_g_y, grids, wts, joint = quadrature.integrate(
        fi, y_i[None, :], n_i[None, :], fixed_arm_dim=arm_idx, fixed_arm_values=ti_rule.pts,
        n_theta=21, return_intermediates=True
    )
    quad_p_ti_g_y /= np.sum(quad_p_ti_g_y * ti_rule.wts, axis=1)[:, None]

    theta_i_sigma = np.sqrt(np.diagonal(-np.linalg.inv(hess), axis1=1, axis2=2))
    theta_i_mu = theta_max
    gaussian_pdf = scipy.stats.norm.pdf(
        ti_rule.pts[:, None],
        theta_i_mu[None, :, arm_idx],
        theta_i_sigma[None, :, arm_idx],
    )
    gaussian_p_ti_g_y = np.sum(
        gaussian_pdf * sigma2_post * sigma2_rule.wts[None, :], axis=1
    )

    plt.plot(ti_rule.pts, gaussian_p_ti_g_y, "r-o", markersize=3, label="INLA-Gaussian")
    plt.plot(ti_rule.pts, quad_p_ti_g_y[0], "b-o", markersize=3, label="Quad")
    plt.legend()
    plt.show()
```

## 2.2: Laplace approximation of $p(\theta_i|y, \sigma^2)$

Instead of assuming $p(\theta_i|y, \sigma^2)$ is Gaussian, we will compute it using the same Laplace approximation method that we used to compute $p(\sigma^2|y)$. That is:
* we optimize over $\mathbf{\theta}_{-i}$ - the list of $\theta$ indices that are not equal to $i$.
* Then we assume that $p(\mathbf{\theta}_{-i})$ is 


```python
y_tiled = np.tile(y_i[None,:], (ti_rule.pts.shape[0], 1))
n_tiled = np.tile(n_i[None,:], (ti_rule.pts.shape[0], 1))
ti_pts_tiled = np.tile(ti_rule.pts[:, None], (1, fi.sigma2_n)) 

ti_max, hess_inv = fi.optimize_mode(y_tiled, n_tiled, fixed_arm_dim=arm_idx, fixed_arm_values=ti_pts_tiled)
ti_logjoint = fi.log_joint(y_tiled, n_tiled, ti_max)

ti_post = np.exp(ti_logjoint + 0.5 * np.log(np.linalg.det(-hess_inv)))
ti_post /= np.sum(ti_post * ti_rule.wts[:, None], axis=0)
laplace_p_ti_g_y = np.sum(ti_post * sigma2_post * sigma2_rule.wts, axis=1)
```

```python
plt.figure(figsize=(7, 4))
plt.plot(ti_rule.pts, gaussian_p_ti_g_y, "r-o", markersize=3, label="INLA-Gaussian")
plt.plot(ti_rule.pts, laplace_p_ti_g_y, "k-o", markersize=3, label="INLA-Laplace")
plt.plot(ti_rule.pts, quad_p_ti_g_y[0], "b-o", markersize=3, label="Quad")
plt.legend()
plt.show()
```

# Section 3: Treat a single arm directly?

What if we *always* analytically integrate arm 0? Both in the computation of $p(\sigma^2|y)$ and in computing $p(\theta_0|y)$?

This is actually quite easy given the calculations already done above. 

```python
new = np.sum(
    np.exp(ti_logjoint - 0.5 * np.log(np.linalg.det(-hess_inv))) * ti_rule.wts[:, None],
    axis=0,
)
new /= np.sum(new * sigma2_rule.wts)
plt.plot(np.log10(sigma2_rule.pts), new * sigma2_rule.wts, label="analytical t0 inla")
plt.plot(np.log10(sigma2_rule.pts), sigma2_post * sigma2_rule.wts, label="simple inla")
plt.legend()
plt.show()
```

```python
t0_analytical_p_ti_g_y = np.sum(ti_post * new * sigma2_rule.wts, axis=1)
```

```python

```
