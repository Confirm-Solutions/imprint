---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: SageMath 9.4
    language: sage
    name: sagemath-9.4
---

```sage
from sage.all import *
from scipy.special import logit
```

```sage
n_arms = 2
ts = [var(f't_{i}') for i in range(n_arms)]
mu_0 = -1.34
mu_0 = var('mu_0')
logit_p1 = logit(0.3)
logit_p1 = var('L')

Q = [[var(f'Q_{i}{j}') for j in range(n_arms)] for i in range(n_arms)]

y = [var(f'y_{i}') for i in range(4)]
n = [var(f'n_{i}') for i in range(4)]
```

```sage
sig2 = var('S')
cov = [[100 for i in range(4)] for j in range(4)]
for i in range(4):
    cov[i][i] += sig2
```

```sage
Qrough = matrix(cov).inverse()
Q = [[Qrough[i][j].full_simplify() for i in range(4)] for j in range(4)]
Q
```

```sage
theta_m0 = [t - mu_0 for t in ts]
theta_adj = [t + logit_p1 for t in ts]
exp_theta_adj = [exp(t) for t in theta_adj]
quad_term = sum([sum([theta_m0[i] * Q[i][j] * theta_m0[j] / 2 for j in range(n_arms)]) for i in range(n_arms)]).full_simplify()
```

```sage
quad_term
```

```sage
bin_term = sum([theta_adj[i] * y[i] - n[i] * log(exp_theta_adj[i] + 1) for i in range(n_arms)]).full_simplify()
```

```sage
bin_term
```

```sage
jll = (quad_term + bin_term)
```

```sage
M = var('M')
nt = 2
full_taylor = jll
for i in range(n_arms):
    full_taylor = full_taylor.taylor(ts[i], M, nt).full_simplify()
```

```sage
full_taylor
```

```sage
A, B = var('A', 'B')
integral = exp(full_taylor)
for i in range(1):
    integral = integrate(integral, ts[i], A, B).full_simplify()
integral
```

```sage
integrate(integral, ts[1], A, B).full_simplify()
```

```sage

```
