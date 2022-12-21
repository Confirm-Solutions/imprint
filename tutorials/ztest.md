# Exploring a one sample Z-Test

A basic introduction to `imprint`.

```python
from imprint.nb_util import setup_nb

# setup_nb is a handy function for setting up some nice plotting defaults.
setup_nb()
import scipy.stats
import matplotlib.pyplot as plt

import imprint as ip
from imprint.models.ztest import ZTest1D
```

```python
import numpy as np

xs = np.linspace(-5, 5, 100)
pdf = scipy.stats.norm.pdf(xs)
plt.plot(xs, pdf)
plt.show()
pdf = scipy.stats.norm.pdf(xs, loc=2)
plt.plot(xs, pdf)
plt.show()
```

```python
plt.plot(xs, scipy.stats.norm.cdf(xs))
plt.axhline(0.975, color="r")
plt.axhline(0.025, color="r")
plt.show()
```

```python
samples = scipy.stats.norm.rvs(loc=true_drug_effect, size=K)
```

```python
np.sum(samples > 2) / len(samples)
```

```python
1 - scipy.stats.norm.cdf(2)
```

```python
100 * np.sum(samples > 1.96) / len(samples)
```

```python
def apply_statistical_test(true_drug_effect, K=10000):
    samples = scipy.stats.norm.rvs(loc=true_drug_effect, size=K)
    number_of_rejections = np.sum(samples > 1.96)
    positive_rate = number_of_rejections / samples.shape[0]
    return positive_rate
```

```python
true_effect = np.linspace(-2, 2, 8)
positive_rate = np.array([apply_statistical_test(tf) for tf in true_effect])
```

```python
plt.plot(true_effect, 100 * positive_rate, "ko")
plt.ylabel(r"Positive Rate (\%)")
plt.xlabel("True Effect")
plt.show()
```

```python
positive_rate = np.array([apply_statistical_test(0) for i in range(100)])
plt.hist(positive_rate, bins=20)
plt.show()
```

```python
g = ip.cartesian_grid(
    theta_min=[-1], theta_max=[1], n=[100], null_hypos=[ip.hypo("x < 0")]
)
# lam = -1.96 because we negated the statistics so we can do a less than
# comparison.
lam = -1.96
K = 8192
rej_df = ip.validate(ZTest1D, g, lam, K=K)
```

```python
g.df.head()
```

```python
rej_df.head()
```

```python
g_rej = g.add_cols(rej_df)
g_rej.df.sort_values("theta0", inplace=True)
true_err = 1 - scipy.stats.norm.cdf(-g_rej.get_theta()[:, 0] - lam)

plt.plot(
    g_rej.df["theta0"],
    100 * g_rej.df["tie_est"],
    "k--o",
    markersize=2,
    label="Monte Carlo estimate",
)
plt.plot(
    g_rej.df["theta0"],
    100 * g_rej.df["tie_cp_bound"],
    "b--o",
    markersize=2,
    label="Clopper-Pearson Bound",
)
plt.plot(
    g_rej.df["theta0"],
    100 * g_rej.df["tie_bound"],
    "r--o",
    markersize=2,
    label="Tilt Bound",
)
plt.plot(
    g_rej.df["theta0"],
    100 * true_err,
    "r-*",
    linewidth=2.5,
    markersize=2,
    label="True Type I Error",
)
plt.axhline(2.5, color="k")
plt.axvline(0, color="k")
plt.ylim([0, 2.6])
plt.legend(fontsize=11, bbox_to_anchor=(0.05, 0.94), loc="upper left")
plt.xlabel("$z$")
_ = plt.ylabel(r"Type I Error (\%)")
```

inputs:
- the experimental design
- the decision rule
- the number of simulations to run
- the values of the statistical
- validation: the decision rule threshold
- calibration: the goal false positive rate

output:
- validation: the upper bound on false positive rate 
- calibration: the tuning parameter value which yield guaranteed goal false positive rate
