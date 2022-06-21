---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: 'Python 3.10.2 64-bit (''imprint'': conda)'
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import scipy.stats
import scipy.optimize
import scipy.integrate
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
```

```python
np.random.seed(100)

n_patients_per_group = 50
n_arms = 4
n_sims = 1000

# The group effects are drawn from a distribution with mean 0.5 and variance 1.0
mean_effect = 0.5
effect_var = 1.0
t_i = scipy.stats.norm.rvs(mean_effect, np.sqrt(effect_var), size=(n_sims, n_arms))

# inverse logit to get probabilities from linear predictors.
p_i = scipy.special.expit(t_i)

n_i = np.full_like(p_i, n_patients_per_group)

# draw actual trial results.
y_i = scipy.stats.binom.rvs(n_patients_per_group, p_i)
y_i.shape
```

## INLA

```python
%%time
import model

post_theta, logpost_theta_data = model.calc_posterior_theta(y_i, n_i)
```

```python
%%time
mu_post, sigma_post = model.calc_posterior_x(post_theta, logpost_theta_data)
```

```python
a_grid = logpost_theta_data["a_grid"]
q_grid = logpost_theta_data["q_grid"]
plt.figure(figsize=(12, 8))
for i in range(6):
    field = post_theta[i]
    levels = None
    plt.subplot(2, 3, i + 1)
    cntf = plt.contourf(a_grid, 1 / q_grid, field.reshape(a_grid.shape), levels=levels)
    plt.contour(
        a_grid,
        1 / q_grid,
        field.reshape(a_grid.shape),
        colors="k",
        linestyles="-",
        linewidths=0.5,
        levels=levels,
    )
    cbar = plt.colorbar(cntf)
    plt.xlabel("$a$")
    plt.ylabel("$1/Q_v$")
plt.show()
```

```python
map_idx = np.argmax(post_theta.reshape((n_sims, -1)), axis=1)
map_A = a_grid.ravel()[map_idx]
map_Q = q_grid.ravel()[map_idx]
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.hist(map_A)
plt.xlabel("a")
plt.subplot(1, 2, 2)
plt.hist(1 / map_Q)
plt.xlabel("1/Q")
plt.show()
```

```python
ci025 = mu_post - 1.96 * sigma_post
ci975 = mu_post + 1.96 * sigma_post
good = (ci025 < t_i) & (t_i < ci975)
np.sum(good) / (n_sims * n_arms)
```

```python
sorted_idxs = np.argsort(t_i[:, 0])
plt.plot(ci025[sorted_idxs, 0])
plt.plot(t_i[sorted_idxs, 0])
plt.plot(mu_post[sorted_idxs, 0])
plt.plot(ci975[sorted_idxs, 0])
plt.show()
```

## MCMC

```python
%%time
from model import mcmc

mcmc_results = mcmc(y_i, n_i, iterations=50000, burn_in=500, skip=3)
assert np.all(
    (mcmc_results["CI025"] < mcmc_results["mean"])
    & (mcmc_results["mean"] < mcmc_results["CI975"])
)
```

```python
print("mcmc results")
effect_estimates_in_cis = (mcmc_results["CI025"][:, :4] < t_i) & (
    t_i < mcmc_results["CI975"][:, :4]
)
mean_est_in_cis = (mcmc_results["CI025"][:, 4] < mean_effect) & (
    mean_effect < mcmc_results["CI975"][:, 4]
)
var_est_in_cis = (mcmc_results["CI025"][:, 5] < effect_var) & (
    effect_var < mcmc_results["CI975"][:, 5]
)
np.sum(effect_estimates_in_cis) / (n_sims * n_arms), np.sum(mean_est_in_cis) / (
    n_sims
), np.sum(var_est_in_cis) / (n_sims)
```

# Profiling MCMC to understand the slow parts

```python
%load_ext line_profiler
from model import proposal, calc_log_joint, calc_log_prior

%lprun -f mcmc -f proposal -f calc_log_joint -f calc_log_prior mcmc(y_i, n_i, iterations=10000, burn_in=500, skip=3)
```
