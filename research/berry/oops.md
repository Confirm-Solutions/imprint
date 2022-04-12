---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3.10.2 ('kevlar')
    language: python
    name: python3
---

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x = np.linspace(-10, 10, 100)
spx = sp.var('spx')
spy = sp.exp(-(spx + 2) ** 2)

y = sp.lambdify(spx, spy)(x)
plt.plot(x, y)
plt.show()
```

```python
grad = spy.diff()
hess = grad.diff()
sp.lambdify(spx, hess)(x)
```

```python

```
