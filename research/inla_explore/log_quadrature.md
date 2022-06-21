---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.10.2 ('imprint')
    language: python
    name: python3
---

```python
import matplotlib.pyplot as plt

%matplotlib inline
%config InlineBackend.figure_format='retina'

%load_ext autoreload
%autoreload 2

import scipy.stats
import numpy as np
import inla
```

```python
A = np.log(1e-8)
B = 0
N = 90
p, w = inla.gauss_rule(N, a=A, b=B)
p10 = np.exp(p)
alpha = 0.0005
beta = 0.000005
f = scipy.stats.invgamma.pdf(p10, alpha, scale=beta)
plt.plot(p, f)
plt.show()
exact = scipy.stats.invgamma.cdf(
    np.exp(B), alpha, scale=beta
) - scipy.stats.invgamma.cdf(np.exp(A), alpha, scale=beta)
est = np.sum(f * np.exp(p) * w)
plt.plot(p, f * np.exp(p) * w)
plt.show()
print(exact, est, est - exact)
```

```python
p, w = inla.gauss_rule(3000, a=np.exp(A), b=np.exp(B))
f = scipy.stats.invgamma.pdf(p, alpha, scale=beta)
exact = scipy.stats.invgamma.cdf(
    np.exp(B), alpha, scale=beta
) - scipy.stats.invgamma.cdf(np.exp(A), alpha, scale=beta)
est = np.sum(f * w)
est, exact, est - exact
```

```python

```
