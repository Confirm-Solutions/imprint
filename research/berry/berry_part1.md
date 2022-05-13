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

See the `intro_to_inla.ipynb` notebook for an introduction to the methods and model used here.

```python
import matplotlib.pyplot as plt

%matplotlib inline
%config InlineBackend.figure_format='retina'

%load_ext autoreload
%autoreload 2

import scipy.stats
import numpy as np

import berry
import util

import inla
import dirty_bayes
import quadrature
import mcmc
```

```python
b = berry.Berry(sigma2_n=90, sigma2_bounds=(1e-8, 1e3))

# I got this data by deconstructing the graphs in in Figure 1 of Berry et al 2013.
n_i = np.array([[i] * 4 for i in [10, 15, 20, 25, 30, 35]])
y_i = np.array(
    [
        [1, 6, 3, 3],
        [3, 8, 5, 4],
        [6, 9, 7, 5],
        [7, 10, 8, 7],
        [8, 10, 9, 8],
        [11, 11, 10, 9],
    ]
)
data = np.stack((y_i, n_i), axis=2)
```

```python
# Run all four methods!
post_hyper, report = inla.calc_posterior_hyper(b, data)
inla_stats = inla.calc_posterior_x(post_hyper, report, b.suc_thresh)
```

```python
db_stats = dirty_bayes.calc_dirty_bayes(
    y_i, n_i, b.mu_0, b.logit_p1, b.suc_thresh, b.sigma2_rule
)
quad_stats = quadrature.quadrature_posterior_theta(b, data, b.suc_thresh)
mcmc_stats = mcmc.mcmc_berry(b, data, b.suc_thresh)
```

```python
berry.figure1_plot(b, "INLA", data, inla_stats)
berry.figure1_plot(b, "DB", data, db_stats)
berry.figure1_plot(b, "Quad", data, quad_stats)
berry.figure1_plot(b, "MCMC", data, mcmc_stats)
```

```python
look_idx = 5
plt.figure(figsize=(10, 10), constrained_layout=True)
for arm_idx in range(4):
    # Quadrature
    ti_rule = util.simpson_rule(61, -2.0, 1.5)
    integrate_dims = [0, 1, 2, 3]
    integrate_dims.remove(arm_idx)
    quad_p_ti_g_y = quadrature.integrate(
        b,
        data[None, look_idx],
        integrate_sigma2=True,
        integrate_thetas=integrate_dims,
        fixed_dims={arm_idx: ti_rule},
        n_theta=9,
    )
    quad_p_ti_g_y /= np.sum(quad_p_ti_g_y * ti_rule.wts, axis=1)[:, None]

    # MCMC
    mcmc_thetai = np.asarray(
        mcmc_data[look_idx].get_samples(False)["theta"][:, arm_idx]
    )
    domain = (np.min(ti_rule.pts), np.max(ti_rule.pts))
    counts, bin_edges = np.histogram(mcmc_thetai, bins=np.linspace(*domain, 51))
    counts = counts.astype(np.float64)
    counts /= np.sum(counts)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    mcmc_pdf = (counts / np.sum(counts)) / bin_widths
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # INLA
    n_arms = 4
    x_mu = report["x0"].reshape((*post_hyper.shape, n_arms))
    x_sigma2 = (
        report["model"].sigma2_from_H(report["H"]).reshape((*post_hyper.shape, n_arms))
    )
    x_sigma = np.sqrt(x_sigma2)
    inla_pdf = scipy.stats.norm.pdf(
        ti_rule.pts[:, None],
        x_mu[None, look_idx, :, arm_idx],
        x_sigma[None, 5, :, arm_idx],
    )
    inla_p_ti_g_y = np.sum(
        inla_pdf * post_hyper[None, look_idx, :] * b.sigma2_rule.wts[None, :], axis=1
    )

    # DB
    x_mu = db_stats["mu_posterior"]
    x_sigma = np.sqrt(db_stats["sigma2_posterior"])
    db_pdf = scipy.stats.norm.pdf(
        ti_rule.pts[:, None],
        x_mu[None, look_idx, :, arm_idx],
        x_sigma[None, 5, :, arm_idx],
    )
    db_p_ti_g_y = np.sum(
        db_pdf
        * db_stats["sigma2_given_y"][None, look_idx, :]
        * b.sigma2_rule.wts[None, :],
        axis=1,
    )

    plt.title(f"Arm {arm_idx}")
    plt.subplot(2, 2, arm_idx + 1)
    plt.plot(ti_rule.pts, db_p_ti_g_y, "m-o", markersize=3, label="DB")
    plt.plot(ti_rule.pts, inla_p_ti_g_y, "r-o", markersize=3, label="INLA")
    plt.plot(ti_rule.pts, quad_p_ti_g_y[0], "b-o", markersize=3, label="Quad")
    plt.plot(bin_centers, mcmc_pdf, "ko", label="MCMC")
    plt.xlabel(f"$\\theta_{arm_idx}$")
    plt.ylabel(f"p($\\theta_{arm_idx}$|y)")
    plt.legend()
plt.show()
```

```python
look_idx = 5
arm_idx = 0
plt.figure(figsize=(15, 10), constrained_layout=True)

# DB
ti_rule = util.simpson_rule(61, -6.0, 2.0)
x_mu = db_stats["mu_posterior"]
x_sigma = np.sqrt(db_stats["sigma2_posterior"])
db_pdf = scipy.stats.norm.pdf(
    ti_rule.pts[:, None],
    x_mu[None, look_idx, :, arm_idx],
    x_sigma[None, 5, :, arm_idx],
)

TT, log_sigma_grid = np.meshgrid(
    ti_rule.pts, np.log10(b.sigma2_rule.pts), indexing="ij"
)
field = db_pdf
levels = None

plt.subplot(2, 3, 1)
plt.title(r"DB $p(\theta_0|\sigma^2,y)$")
cntf = plt.contourf(TT, log_sigma_grid, field, levels=levels, extend="both")
plt.contour(
    TT,
    log_sigma_grid,
    field,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)

plt.subplot(2, 3, 2)
plt.title("DB weights")
plt.plot(
    np.log10(b.sigma2_rule.pts),
    db_stats["sigma2_given_y"][look_idx] * b.sigma2_rule.wts,
)

plt.subplot(2, 3, 3)
plt.title(r"DB $p(\theta_0|\sigma^2,y) * p(\sigma^2|y) * d\sigma^2$")
field = (
    db_pdf * db_stats["sigma2_given_y"][None, look_idx, :] * b.sigma2_rule.wts[None, :]
)
cntf = plt.contourf(TT, log_sigma_grid, field, levels=levels, extend="both")
plt.contour(
    TT,
    log_sigma_grid,
    field,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\sigma^2$")

# INLA
ti_rule = util.simpson_rule(61, -6.0, 2.0)
n_arms = 4
x_mu = report["x0"].reshape((*post_hyper.shape, n_arms))
x_sigma2 = (
    report["model"].sigma2_from_H(report["H"]).reshape((*post_hyper.shape, n_arms))
)
x_sigma = np.sqrt(x_sigma2)
inla_pdf = scipy.stats.norm.pdf(
    ti_rule.pts[:, None],
    x_mu[None, look_idx, :, arm_idx],
    x_sigma[None, 5, :, arm_idx],
)

field = inla_pdf
plt.subplot(2, 3, 4)
plt.title(r"INLA $p(\theta_0|\sigma^2,y)$")
cntf = plt.contourf(TT, log_sigma_grid, field, levels=levels, extend="both")
plt.contour(
    TT,
    log_sigma_grid,
    field,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)

plt.subplot(2, 3, 5)
plt.title("INLA weights")
plt.plot(np.log10(b.sigma2_rule.pts), post_hyper[look_idx] * b.sigma2_rule.wts)

plt.subplot(2, 3, 6)
plt.title(r"INLA $p(\theta_0|\sigma^2,y) * p(\sigma^2|y) * d\sigma^2$")
field = inla_pdf * post_hyper[None, look_idx, :] * b.sigma2_rule.wts[None, :]
cntf = plt.contourf(TT, log_sigma_grid, field, levels=levels, extend="both")
plt.contour(
    TT,
    log_sigma_grid,
    field,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\sigma^2$")

plt.show()
```

# Figure 2

```python
# I got this data by deconstructing the graphs in in Figure 1 of Berry et al 2013.
n_i = np.array(
    [
        [10, 10, 10, 10],
        [15, 15, 15, 15],
        [20, 20, 20, 20],
        [20, 20, 25, 25],
        [20, 20, 30, 30],
        [20, 20, 35, 35],
    ]
)
y_i = np.array(
    [
        [0, 1, 3, 3],
        [0, 1, 4, 5],
        [0, 1, 6, 6],
        [0, 1, 6, 7],
        [0, 1, 7, 8],
        [0, 1, 9, 10],
    ],
    dtype=np.float64,
)
data = np.stack((y_i, n_i), axis=2)

y_i_no0 = y_i.copy()
y_i_no0[y_i_no0 == 0] = 0.5
data_no0 = np.stack((y_i_no0, n_i), axis=2)

post_hyper, report = inla.calc_posterior_hyper(b, data_no0)
inla_stats = inla.calc_posterior_x(post_hyper, report, b.suc_thresh)
db_stats = dirty_bayes.calc_dirty_bayes(
    y_i_no0, n_i, b.mu_0, b.logit_p1, b.suc_thresh, b.sigma2_rule
)
quad_stats = quadrature.quadrature_posterior_theta(b, data, b.suc_thresh)
mcmc_data, mcmc_stats = mcmc.mcmc_berry(b, data, b.suc_thresh)
```

```python
berry.figure1_plot(b, "INLA", data, inla_stats)
berry.figure1_plot(b, "DB", data, db_stats)
berry.figure1_plot(b, "Quad", data, quad_stats)
berry.figure1_plot(b, "MCMC", data, mcmc_stats)
```

```python
fig = plt.figure(figsize=(10, 10))
plt.suptitle("Final analysis")
outergs = fig.add_gridspec(2, 2, hspace=0.3)
innergs = outergs[0].subgridspec(2, 1, wspace=0, hspace=0, height_ratios=[0.7, 0.3])
berry.figure1_subplot(innergs[0], innergs[1], 5, b, data, inla_stats, title="INLA")
innergs = outergs[1].subgridspec(2, 1, wspace=0, hspace=0, height_ratios=[0.7, 0.3])
berry.figure1_subplot(innergs[0], innergs[1], 5, b, data, db_stats, title="DB")
innergs = outergs[2].subgridspec(2, 1, wspace=0, hspace=0, height_ratios=[0.7, 0.3])
berry.figure1_subplot(
    innergs[0], innergs[1], 5, b, data, quad_stats, title="Quadrature"
)
innergs = outergs[3].subgridspec(2, 1, wspace=0, hspace=0, height_ratios=[0.7, 0.3])
berry.figure1_subplot(innergs[0], innergs[1], 5, b, data, mcmc_stats, title="MCMC")
```

## Looking at $p(\theta_i|y_k)$

```python
y_i_no0 = y_i.copy()
y_i_no0[y_i_no0 == 0] = 0.00001
data_no0 = np.stack((y_i_no0, n_i), axis=2)
db_stats = dirty_bayes.calc_dirty_bayes(
    y_i_no0, n_i, b.mu_0, b.logit_p1, b.suc_thresh, b.sigma2_rule
)
```

```python
post_hyper, report = inla.calc_posterior_hyper(b, data)
inla_stats = inla.calc_posterior_x(post_hyper, report, b.suc_thresh)
```

```python
look_idx = 5
plt.figure(figsize=(10, 10), constrained_layout=True)
for arm_idx in range(4):
    # Quadrature
    ti_rule = util.simpson_rule(61, -6.0, 2.0)
    integrate_dims = [0, 1, 2, 3]
    integrate_dims.remove(arm_idx)
    quad_p_ti_g_y = quadrature.integrate(
        b,
        data[None, look_idx],
        integrate_sigma2=True,
        integrate_thetas=integrate_dims,
        fixed_dims={arm_idx: ti_rule},
        n_theta=9,
    )
    quad_p_ti_g_y /= np.sum(quad_p_ti_g_y * ti_rule.wts, axis=1)[:, None]

    # MCMC
    mcmc_thetai = np.asarray(
        mcmc_data[look_idx].get_samples(False)["theta"][:, arm_idx]
    )
    domain = (np.min(ti_rule.pts), np.max(ti_rule.pts))
    counts, bin_edges = np.histogram(mcmc_thetai, bins=np.linspace(*domain, 51))
    counts = counts.astype(np.float64)
    counts /= np.sum(counts)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    mcmc_pdf = (counts / np.sum(counts)) / bin_widths
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # INLA
    n_arms = 4
    x_mu = report["x0"].reshape((*post_hyper.shape, n_arms))
    x_sigma2 = (
        report["model"].sigma2_from_H(report["H"]).reshape((*post_hyper.shape, n_arms))
    )
    x_sigma = np.sqrt(x_sigma2)
    inla_pdf = scipy.stats.norm.pdf(
        ti_rule.pts[:, None],
        x_mu[None, look_idx, :, arm_idx],
        x_sigma[None, 5, :, arm_idx],
    )
    inla_p_ti_g_y = np.sum(
        inla_pdf * post_hyper[None, look_idx, :] * b.sigma2_rule.wts[None, :], axis=1
    )

    # DB
    x_mu = db_stats["mu_posterior"]
    x_sigma = np.sqrt(db_stats["sigma2_posterior"])
    db_pdf = scipy.stats.norm.pdf(
        ti_rule.pts[:, None],
        x_mu[None, look_idx, :, arm_idx],
        x_sigma[None, 5, :, arm_idx],
    )
    db_p_ti_g_y = np.sum(
        db_pdf
        * db_stats["sigma2_given_y"][None, look_idx, :]
        * b.sigma2_rule.wts[None, :],
        axis=1,
    )

    plt.title(f"Arm {arm_idx}")
    plt.subplot(2, 2, arm_idx + 1)
    plt.plot(ti_rule.pts, db_p_ti_g_y, "m-o", markersize=3, label="DB")
    plt.plot(ti_rule.pts, inla_p_ti_g_y, "r-o", markersize=3, label="INLA")
    plt.plot(ti_rule.pts, quad_p_ti_g_y[0], "b-o", markersize=3, label="Quad")
    plt.plot(bin_centers, mcmc_pdf, "ko", label="MCMC")
    plt.xlabel(f"$\\theta_{arm_idx}$")
    plt.ylabel(f"p($\\theta_{arm_idx}$|y)")
    plt.legend()
plt.show()
```

```python
look_idx = 5
arm_idx = 2
plt.figure(figsize=(15, 10), constrained_layout=True)

# DB
ti_rule = util.simpson_rule(61, -6.0, 2.0)
x_mu = db_stats["mu_posterior"]
x_sigma = np.sqrt(db_stats["sigma2_posterior"])
db_pdf = scipy.stats.norm.pdf(
    ti_rule.pts[:, None],
    x_mu[None, look_idx, :, arm_idx],
    x_sigma[None, 5, :, arm_idx],
)

TT, log_sigma_grid = np.meshgrid(
    ti_rule.pts, np.log10(b.sigma2_rule.pts), indexing="ij"
)
field = db_pdf
levels = None

plt.subplot(2, 3, 1)
plt.title(r"DB $p(\theta_0|\sigma^2,y)$")
cntf = plt.contourf(TT, log_sigma_grid, field, levels=levels, extend="both")
plt.contour(
    TT,
    log_sigma_grid,
    field,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)

plt.subplot(2, 3, 2)
plt.title("DB weights")
plt.plot(
    np.log10(b.sigma2_rule.pts),
    db_stats["sigma2_given_y"][look_idx] * b.sigma2_rule.wts,
)

plt.subplot(2, 3, 3)
plt.title(r"DB $p(\theta_0|\sigma^2,y) * p(\sigma^2|y) * d\sigma^2$")
field = (
    db_pdf * db_stats["sigma2_given_y"][None, look_idx, :] * b.sigma2_rule.wts[None, :]
)
cntf = plt.contourf(TT, log_sigma_grid, field, levels=levels, extend="both")
plt.contour(
    TT,
    log_sigma_grid,
    field,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\sigma^2$")

# INLA
ti_rule = util.simpson_rule(61, -6.0, 2.0)
n_arms = 4
x_mu = report["x0"].reshape((*post_hyper.shape, n_arms))
x_sigma2 = (
    report["model"].sigma2_from_H(report["H"]).reshape((*post_hyper.shape, n_arms))
)
x_sigma = np.sqrt(x_sigma2)
inla_pdf = scipy.stats.norm.pdf(
    ti_rule.pts[:, None],
    x_mu[None, look_idx, :, arm_idx],
    x_sigma[None, 5, :, arm_idx],
)

field = inla_pdf
plt.subplot(2, 3, 4)
plt.title(r"INLA $p(\theta_0|\sigma^2,y)$")
cntf = plt.contourf(TT, log_sigma_grid, field, levels=levels, extend="both")
plt.contour(
    TT,
    log_sigma_grid,
    field,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)

plt.subplot(2, 3, 5)
plt.title("INLA weights")
plt.plot(np.log10(b.sigma2_rule.pts), post_hyper[look_idx] * b.sigma2_rule.wts)

plt.subplot(2, 3, 6)
plt.title(r"INLA $p(\theta_0|\sigma^2,y) * p(\sigma^2|y) * d\sigma^2$")
field = inla_pdf * post_hyper[None, look_idx, :] * b.sigma2_rule.wts[None, :]
cntf = plt.contourf(TT, log_sigma_grid, field, levels=levels, extend="both")
plt.contour(
    TT,
    log_sigma_grid,
    field,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\sigma^2$")

plt.show()
```

```python
p_sigma_g_y = quadrature.integrate(b, data[[0, 1, 5]], integrate_thetas=(0, 1, 2, 3))
p_sigma_g_y /= np.sum(p_sigma_g_y * b.sigma2_rule.wts, axis=1)[:, None]
# TODO: would be cool to do a CDF here using a generated product rule for each subset of points.
plt.plot(np.log10(b.sigma2_rule.pts), p_sigma_g_y[0], "b", label="interim 1")
plt.plot(np.log10(b.sigma2_rule.pts), p_sigma_g_y[1], "r", label="interim 2")
plt.plot(np.log10(b.sigma2_rule.pts), p_sigma_g_y[2], "k", label="look 6 (final)")
plt.legend()
plt.show()
```

```python
t0_rule = util.simpson_rule(61, -6.0, 1.0)
p_t0_g_y = quadrature.integrate(
    b,
    data[[0, 1, 5]],
    integrate_sigma2=True,
    integrate_thetas=(1, 2, 3),
    fixed_dims={0: t0_rule},
)
p_t0_g_y /= np.sum(p_t0_g_y * t0_rule.wts, axis=1)[:, None]

plt.plot(t0_rule.pts, p_t0_g_y[0], "b-o", markersize=3, label="interim 1")
plt.plot(t0_rule.pts, p_t0_g_y[1], "r-o", markersize=3, label="interim 2")
plt.plot(t0_rule.pts, p_t0_g_y[2], "k-o", markersize=3, label="look 6 (final)")
plt.legend()
plt.ylabel(r"p($\theta_0$ | y)")
plt.xlabel(r"$\theta_0$")
plt.show()
```

```python
theta_map = t0_rule.pts[np.argmax(p_t0_g_y, axis=1)]
theta_map
```

```python
t0_rule.pts[:3], t0_rule.wts[:3]
```

```python
cdf = []
cdf_pts = []
for i in range(3, t0_rule.pts.shape[0], 2):
    # Note that t0_rule.wts[:i] will be different from cdf_rule.wts!!
    cdf_rule = util.simpson_rule(i, t0_rule.pts[0], t0_rule.pts[i - 1])
    cdf.append(np.sum(p_t0_g_y[:, :i] * cdf_rule.wts[:i], axis=1))
    cdf_pts.append(t0_rule.pts[i - 1])
cdf = np.array(cdf)
cdf_pts = np.array(cdf_pts)
```

```python
cilow = cdf_pts[np.argmax(cdf > 0.025, axis=0)]
cihi = cdf_pts[np.argmax(cdf > 0.975, axis=0)]
cilow, cihi
```

```python
b.suc_thresh
```

```python
plt.plot(cdf_pts, cdf[:, 0], "b-o", markersize=3, label="interim 1")
plt.plot(cdf_pts, cdf[:, 1], "r-o", markersize=3, label="interim 2")
plt.plot(cdf_pts, cdf[:, 2], "k-o", markersize=3, label="look 6 (final)")
plt.legend()
plt.ylabel(r"F($\theta_0$ | y)")
plt.xlabel(r"$\theta_0$")
plt.show()
```

```python
plt.figure(figsize=(8, 8), constrained_layout=True)
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.title(f"Arm {i+1}")
    inla_graph = report["x0"][5, :, i]
    db_graph = db_stats["mu_posterior"][5, :, i]
    plt.plot(np.log10(b.sigma2_rule.pts), db_graph, label="DB")
    plt.plot(np.log10(b.sigma2_rule.pts), inla_graph, label="INLA")
    plt.ylabel(r"Mean of p($\theta_{i} | y, \sigma^2$)")
    plt.xlabel("$log_{10} \sigma^2$")
    plt.legend()
plt.show()
```

### Skewness in the marginals

Why is the confidence interval on the 0-th arm in the figure above so large? This is a case where one of the core INLA assumptions breaks down. INLA assumes that p(x|y,\theta) is approximately normal. In this particular case, that assumption is not correct. Intuitively, with 0 successes out of 20 patients, there is a lot more potential for small $x_0$ values than potential for large $x_0$ values. As you can see below, there is substantial skewness. There are approaches to deal with this. See here: https://github.com/mikesklar/kevlar/issues/15

```python
x0_vs = np.linspace(-15, 5, 100)
x123_vs = np.full_like(x0_vs, -1.0)
x = np.array([x0_vs, x123_vs, x123_vs, x123_vs]).T.copy()
lj = model.log_joint(model, x, data[0], np.array([[-1.0, 10.0]]))
plt.plot(x0_vs, np.exp(lj))
plt.show()
```

```python
mu_post, sigma_post = inla.calc_posterior_x(post_theta2, report2)

# expit(mu_post) is the posterior estimate of the mean probability.
p_post = scipy.special.expit(mu_post)

# two sigma confidence intervals transformed from logit to probability space.
cilow = scipy.special.expit(mu_post - 2 * sigma_post)
cihigh = scipy.special.expit(mu_post + 2 * sigma_post)
```

```python
cilow[0], cihigh[0]
```

```python
total_sum = np.sum(np.exp(lj))
mean = x0_vs[np.argmax(np.exp(lj))]
ci025 = x0_vs[np.argmax(np.cumsum(np.exp(lj)) / total_sum > 0.05)]
ci975 = x0_vs[np.argmax(np.cumsum(np.exp(lj)) / total_sum > 0.95)]
ci025, ci975, np.abs(mean - ci025), np.abs(mean - ci975), scipy.special.expit(
    ci025
), scipy.special.expit(ci975)
```
