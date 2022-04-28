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

This notebook is verifying that the two hyperparameter $(\mu, \sigma^2)$ version of the Berry problem is producing the same results as the one parameter condensed $\sigma^2$ version.

At the end of the notebook, we plot $p(\sigma^2 | y)$ as determined by INLA via both of these versions of the problem.

```python
%load_ext autoreload
%autoreload 2
import numpy as np
import berry
import inla
import matplotlib.pyplot as plt

y_i = np.array([[3, 8, 5, 4]])
n_i = np.full((1, 4), 15)
data = np.stack((y_i, n_i), axis=2)

n = 90
a, b = (1e-4, 1e-0)
b_mu = berry.BerryMu(sigma2_n=n, sigma2_bounds=(a, b))
post_hyper_mu, report_mu = inla.calc_posterior_hyper(b_mu, data)

b_no_mu = berry.Berry(sigma2_n=n, sigma2_bounds=(a, b))
post_hyper_no_mu, report_no_mu = inla.calc_posterior_hyper(b_no_mu, data)

berry.plot_2d_field(
    report_mu, np.log10(post_hyper_mu[0]), levels=np.linspace(-15, 5, 21)
)
plt.show()

plt.plot(np.log10(b_no_mu.sigma2_rule.pts), report_no_mu["x0"][0, :, :])
plt.show()
```

```python
num_int_mu = np.sum(post_hyper_mu[0] * b_mu.mu_rule.wts[:, None], axis=0)
num_int_mu /= np.sum(num_int_mu * b_mu.sigma2_rule.wts, axis=0)
```

```python
num_int_mu
```

```python
log_sig = np.log10(b_no_mu.sigma2_rule.pts)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(log_sig, np.log10(post_hyper_no_mu[0]), "r-")
plt.plot(log_sig, np.log10(num_int_mu), "k-")
plt.ylim([-15, 5])
plt.subplot(1, 2, 2)
plt.plot(log_sig, (post_hyper_no_mu[0] - num_int_mu) / post_hyper_no_mu[0], "r-")
plt.show()
```

```python
thresh = np.full((1, 4), -0.0)
mu_stats = inla.calc_posterior_x(post_hyper_mu, report_mu, thresh)
no_mu_stats = inla.calc_posterior_x(post_hyper_no_mu, report_no_mu, thresh)
print(mu_stats["exceedance"])
print(no_mu_stats["exceedance"])
```

```python

```

```python

```
