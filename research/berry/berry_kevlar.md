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
import berrylib.util as util

util.setup_nb()

```

```python
from scipy.special import logit, expit
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pyimprint.grid as grid
import berrylib.fast_inla as fast_inla
import berrylib.binomial as binomial
import berrylib.grid as berrylibgrid

```

```python
name = "berry2d"
n_arms = 2
n_arm_samples = 35
seed = 10
n_theta_1d = 32
sim_size = 1000
```

```python

%%time
# null is:
# theta_i <= logit(0.1)
# the normal should point towards the negative direction. but that also
# means we need to negate the logit(0.1) offset
# so, for example, arm 2 out of 4 would be:
# n = [0, 0, -1, 0]
# c = -logit(0.1)
null_hypos = [
    berrylibgrid.HyperPlane(-np.identity(n_arms)[i], -logit(0.1)) for i in range(n_arms)
]
theta1d = [np.linspace(-3.5, 1.0, 2 * n_theta_1d + 1)[1::2] for i in range(n_arms)]
theta = np.stack(np.meshgrid(*theta1d), axis=-1).reshape((-1, len(theta1d)))
radii = np.empty(theta.shape)
for i in range(theta.shape[1]):
    radii[:, i] = 0.5 * (theta1d[i][1] - theta1d[i][0])
g = berrylibgrid.prune(berrylibgrid.build_grid(theta, radii, null_hypos))
```

```python
theta = g.thetas
theta_tiles = g.thetas[g.grid_pt_idx]
```

```python
%%time
fi = fast_inla.FastINLA(n_arms)
rejection_table = binomial.build_rejection_table(n_arms, n_arm_samples, fi.rejection_inference)
```

```python
%%time
np.random.seed(seed)
samples = np.random.uniform(size=(sim_size, n_arm_samples, n_arms))
accumulator = binomial.binomial_accumulator(lambda y,n: binomial.lookup_rejection(rejection_table, y))

# Chunking improves performance dramatically for larger tile counts:
# ~6x for a 64^3 grid
typeI_sum = np.empty(theta_tiles.shape[0])
typeI_score = np.empty((theta_tiles.shape[0], n_arms))
chunk_size = 5000
n_chunks = int(np.ceil(theta_tiles.shape[0] / chunk_size))
for i in range(n_chunks):
    start = i * chunk_size
    end = (i + 1) * chunk_size
    end = min(end, theta_tiles.shape[0])
    typeI_sum[start:end], typeI_score[start:end] = accumulator(theta_tiles[start:end], g.null_truth[start:end], samples)
```

```python
_, n_tiles_per_pt = np.unique(g.grid_pt_idx, return_counts=True)
pos_start = np.zeros(g.n_grid_pts, dtype=np.int64)
pos_start[1:] = np.cumsum(n_tiles_per_pt)[:-1]
is_null_per_arm_gridpt = np.add.reduceat(g.null_truth, pos_start, axis=0) > 0

plt.title("Is null for arm 0?")
plt.scatter(theta[:, 0], theta[:, 1], c=is_null_per_arm_gridpt[:, 0], cmap="Set1")
plt.hlines(logit(0.1), -4, 2)
plt.vlines(logit(0.1), -4, 2)
plt.xlim(np.min(theta[:, 0]) - 0.2, np.max(theta[:, 0]) + 0.2)
plt.ylim(np.min(theta[:, 1]) - 0.2, np.max(theta[:, 1]) + 0.2)
plt.colorbar()
plt.show()

plt.title("Is null for arm 1?")
plt.scatter(theta[:, 0], theta[:, 1], c=is_null_per_arm_gridpt[:, 1], cmap="Set1")
plt.hlines(logit(0.1), -4, 2)
plt.vlines(logit(0.1), -4, 2)
plt.xlim(np.min(theta[:, 0]) - 0.2, np.max(theta[:, 0]) + 0.2)
plt.ylim(np.min(theta[:, 1]) - 0.2, np.max(theta[:, 1]) + 0.2)
plt.colorbar()
plt.show()

plt.title("Tile count per grid point")
plt.scatter(theta[:, 0], theta[:, 1], c=n_tiles_per_pt)
plt.colorbar()
plt.show()

```

```python
plt.figure()
plt.title("Type I error at grid points.")
plt.scatter(theta_tiles[:, 0], theta_tiles[:, 1], c=typeI_sum / sim_size)
cbar = plt.colorbar()
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
cbar.set_label("Type I error")
plt.show()
plt.title("Yellow points are above 10\%")
plt.scatter(theta_tiles[:, 0], theta_tiles[:, 1], c=typeI_sum / sim_size > 0.1)
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.show()

```

```python
corners = g.vertices
c_flat = corners.reshape((-1, 2))
plt.scatter(c_flat[:, 0], c_flat[:, 1], c="k")
plt.scatter(theta[:, 0], theta[:, 1], c="m")
plt.hlines(logit(0.1), -4, 2, "r")
plt.vlines(logit(0.1), -4, 2, "r")
plt.xlim(np.min(theta[:, 0]) - 0.2, np.max(theta[:, 0]) + 0.2)
plt.ylim(np.min(theta[:, 1]) - 0.2, np.max(theta[:, 1]) + 0.2)
plt.show()

```

```python
%%time
tile_radii = g.radii[g.grid_pt_idx]
sim_sizes = np.full(g.n_tiles, sim_size)
total, d0, d0u, d1w, d1uw, d2uw = binomial.upper_bound(
    theta_tiles,
    tile_radii,
    corners,
    sim_sizes,
    n_arm_samples,
    typeI_sum.to_py(),
    typeI_score,
)

```

```python
plt.figure()
plt.title("Type I error at grid points.")
plt.scatter(theta_tiles[:, 0], theta_tiles[:, 1], c=total)
cbar = plt.colorbar()
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
cbar.set_label("Type I error")
plt.show()
```

# Comparing against MCMC for a few grid points

This is a check to determine how much of the type I error is real versus an artifact coming from INLA.



```python
import berrylib.mcmc as mcmc

```

```python
sorted_idxs = np.argsort(typeI_sum)
idx = sorted_idxs[-1]
theta_tiles[idx]

```

```python
# y_mcmc = y[idx, :]
# n_mcmc = np.full((sim_size, n_arms), n_arm_samples)
# data_mcmc = np.stack((y_mcmc, n_mcmc), axis=-1)
# n_mcmc_sims = 1000
# results_mcmc = mcmc.mcmc_berry(
#     data_mcmc[:n_mcmc_sims], fi.logit_p1, np.full(n_mcmc_sims, fi.thresh_theta), n_arms=2
# )
# success_mcmc = results_mcmc["exceedance"] > critical_values[0]

```

```python
import pickle

load = True
if load:
    with open(f"berry_imprint_mcmc{idx}.pkl", "rb") as f:
        results_mcmc = pickle.load(f)
else:
    with open(f"berry_imprint_mcmc{idx}.pkl", "wb") as f:
        pickle.dump(results_mcmc, f)

```

```python
mcmc_typeI = np.sum(
    np.any(success_mcmc & is_null_per_arm[idx, None, :], axis=-1), axis=-1
)
inla_typeI = typeI_sum[idx]
mcmc_typeI, inla_typeI

```

```python
bad_sim_idxs = np.where(
    np.any((success[idx] & (~success_mcmc)) & is_null_per_arm[idx, None], axis=-1)
)[0]
unique_bad = np.unique(y[idx, bad_sim_idxs], axis=0)
print("theta =", theta_tiles[idx])
bad_count = 0
for i in range(unique_bad.shape[0]):
    y_bad = unique_bad[i]
    other_sim_idx = np.where((y[idx, :, 0] == y_bad[0]) & (y[idx, :, 1] == y_bad[1]))[0]
    pct_mcmc = (
        np.any(
            success_mcmc[other_sim_idx] & is_null_per_arm[idx, None, :], axis=-1
        ).sum()
        / other_sim_idx.shape[0]
    )
    if pct_mcmc < 0.2:
        print("bad y =", y_bad, "count =", other_sim_idx.shape[0])
        bad_count += other_sim_idx.shape[0]
print("\ninla type I =", inla_typeI)
print("bad type I count =", bad_count)
print('"true" type I count =', inla_typeI - bad_count)
# print('pct of mcmc sims that had type I error', pct_mcmc)
# print('')

# print(
#     "unique y where INLA says type 1 but MCMC says not type 1: ",
#     np.unique(y[idx, bad_sim_idxs], axis=0),
# )

```

```python
bad_sim_idxs = np.where(
    np.any((success[idx] & (success_mcmc)) & is_null_per_arm[idx, None], axis=-1)
)[0]
print("theta: ", theta_tiles[idx])
print(
    "unique y where INLA says type 1 and MCMC also says type 1: ",
    np.unique(y[idx, bad_sim_idxs], axis=0),
)

```

```python
bad_sim_idxs = np.where(
    np.any((~success[idx] & (~success_mcmc)) & is_null_per_arm[idx, None], axis=-1)
)[0]
print("theta: ", theta_tiles[idx])
print(
    "unique y where INLA says type 1 and MCMC also says type 1: ",
    np.unique(y[idx, bad_sim_idxs], axis=0),
)

```

# Building a rejection table

```python
%%time
rejection_table = binomial.build_rejection_table(n_arms, n_arm_samples, fi.rejection_inference)
np.random.seed(10)
typeI_sum = np.zeros(theta_tiles.shape[0])
typeI_score = np.zeros(theta_tiles.shape)
accumulator = binomial.binomial_accumulator(lambda y,n: binomial.lookup_rejection(rejection_table, y))
for i in range(sim_size):
    samples = np.random.uniform(0, 1, size=(1, n_arm_samples, fi.n_arms))
    chunk_typeI_sum, chunk_typeI_score = accumulator(theta_tiles, is_null_per_arm, samples)
    typeI_sum += chunk_typeI_sum
    typeI_score += chunk_typeI_score
```

```python
typeI_sum.shape, theta_tiles.shape

```

```python
plt.figure(figsize=(8, 8))
plt.scatter(theta_tiles[:, 0], theta_tiles[:, 1], c=typeI_sum / sim_size)
plt.colorbar()
plt.show()

```

# Running a 4D grid

```python
import berrylib.util as util

util.setup_nb()

# NOTE: don't be surprised if this JAX code does not scale linearly.
# JAX already uses parallelism within a single CPU device, so using multiple CPU
# devices will only take advantage of the remaining unused cores. In addition,
# there will be thrashing between the different threads trying to claim the same
# cores. This could be fixed by CPU pinning the JAX threads. There might be more
# potential to do this with a multiprocessing-based version of this
# parallelization. However, I didn't do this at the moment because I'm working
# on Mac where there is are no easy tools for CPU pinning threads. In addition, for
# higher performance, we should really be running this on a GPU where the effort
# of setting up tools for CPU pinning will have been wasted.
# see: https://github.com/google/jax/issues/743
# some other links that might be useful for parallel and multiprocessing JAX:
# - this link provides a path to cpu pinning on mac: http://www.hybridkernel.com/2015/01/18/binding_threads_to_cores_osx.html
# - more info on JAX multithreading: https://github.com/google/jax/issues/1539
# - running JAX inside multiprocessing: https://github.com/google/jax/issues/1805
n_requested_cores = 8
import os

os.environ["XLA_FLAGS"] = (
    f"--xla_force_host_platform_device_count={n_requested_cores} "
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from scipy.special import logit, expit
import matplotlib.pyplot as plt
import numpy as np
from pyimprint import mt19937
import pyimprint.grid as grid
from pyimprint.driver import accumulate_process
import pyimprint.core.model.binomial
import berrylib.binomial as binomial

import berrylib.fast_inla as fast_inla

name = "berry2"
fi = fast_inla.FastINLA(4)
seed = 10
n_arm_samples = 35
n_theta_1d = 10
sim_size = 10000

```

```python
loaded = np.load(f"{name}_{n_theta_1d}_{sim_size}.npy", allow_pickle=True).tolist()
gr = loaded["gr"]
typeI_sum = loaded["sum"]
typeI_score = loaded["score"]
table = loaded["table"]
theta_tiles = loaded["theta_tiles"]
is_null_per_arm = loaded["is_null_per_arm"]

```

```python
%%time
# define null hypos
null_hypos = []
for i in range(fi.n_arms):
    n = np.zeros(fi.n_arms)
    # null is:
    # theta_i <= logit(0.1)
    # the normal should point towards the negative direction. but that also
    # means we need to negate the logit(0.1) offset
    n[i] = -1
    null_hypos.append(grid.HyperPlane(n, -logit(0.1)))

gr = grid.make_cartesian_grid_range(n_theta_1d, np.full(fi.n_arms, -3.5), np.full(fi.n_arms, 1.0), sim_size)
gr.create_tiles(null_hypos)
gr.prune()
theta_tiles = grid.theta_tiles(gr)
is_null_per_arm = grid.is_null_per_arm(gr)
```

```python
%%time
table = binomial.build_rejection_table(fi.n_arms, n_arm_samples, fi.rejection_inference)
```

# Parallel Jax accumulation

```python
import time
import jax
import jax.numpy as jnp

reject_fnc = lambda y, n: binomial.lookup_rejection(table, y, n)
accum = binomial.binomial_accumulator(reject_fnc)
paccum = jax.pmap(accum, in_axes=(None, None, 0), out_axes=(0, 0), axis_name="i")

```

```python
%%time
np.random.seed(seed)
typeI_sum = np.zeros(theta_tiles.shape[0], dtype=np.uint32)
typeI_score = np.zeros(theta_tiles.shape)
n_cores = jax.local_device_count()
for i in range(int(np.ceil(sim_size / n_cores))):
    chunk_size = np.minimum(n_cores, sim_size - i * n_cores)
    samples = np.random.uniform(0, 1, size=(chunk_size, 1, n_arm_samples, fi.n_arms))
    s, c = paccum(theta_tiles, is_null_per_arm, samples)
    typeI_sum += s.sum(axis=0)
    typeI_score += c.sum(axis=0)
```

```python
i * n_cores, sim_size

```

```python
# If you kill the simulation early, run these lines to accept the smaller sim_size
# sim_size = (i + 1) * n_cores
# gr.sim_sizes()[:] = sim_size

```

```python
np.save(
    f"{name}_{n_theta_1d}_{sim_size}.npy",
    dict(
        sum=typeI_sum,
        score=typeI_score,
        gr=gr,
        table=table,
        theta_tiles=theta_tiles,
        is_null_per_arm=is_null_per_arm,
    ),
    allow_pickle=True,
)

```

# Making some figures!

```python
# for t2_idx, t3_idx in [(16, 16), (8, 8)]:
for t2_idx, t3_idx in [(4, 4), (8, 8)]:
    t2 = np.unique(theta_tiles[:, 2])[t2_idx]
    t3 = np.unique(theta_tiles[:, 2])[t3_idx]
    selection = (theta_tiles[:, 2] == t2) & (theta_tiles[:, 3] == t3)

    plt.figure(figsize=(6, 6), constrained_layout=True)
    plt.title(f"slice: ($\\theta_2, \\theta_3) \\approx$ ({t2:.1f}, {t3:.1f})")
    plt.scatter(
        theta_tiles[selection, 0],
        theta_tiles[selection, 1],
        c=typeI_sum[selection] / sim_size,
        s=120,
    )
    cbar = plt.colorbar()
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    cbar.set_label("Fraction of Type I errors")
    plt.savefig(f"grid_type1_{t2_idx}_{t3_idx}.png", dpi=300, bbox_inches="tight")
    plt.show()

```

# Compute the bound and save slices for the Plotly frontend

```python
from pyimprint.model.binomial import SimpleSelection

sys.path.append("../../python/example")
import utils

delta = 0.025
simple_selection_model = SimpleSelection(fi.n_arms, n_arm_samples, 1, [])
simple_selection_model.critical_values([fi.critical_value])

```

```python
from pyimprint.bound import TypeIErrorAccum

acc_o = TypeIErrorAccum(simple_selection_model.n_models(), gr.n_tiles(), gr.n_params())
typeI_sum = typeI_sum.astype(np.uint32).reshape((1, -1))
score_sum = typeI_score.flatten()
acc_o.pool_raw(typeI_sum, score_sum)
print(np.all(acc_o.typeI_sum() == typeI_sum))
print(np.all(acc_o.score_sum() == score_sum))

```

```python
P, B = utils.create_ub_plot_inputs(simple_selection_model, acc_o, gr, delta)

```

```python
np.save(
    f"{name}_{n_theta_1d}_{sim_size}.npy",
    dict(
        P=P,
        B=B,
        sum=typeI_sum,
        score=typeI_score,
        gr=gr,
        table=table,
        theta_tiles=theta_tiles,
        is_null_per_arm=is_null_per_arm,
    ),
    allow_pickle=True,
)

```

```python
for t2_idx, t3_idx in [(40, 40), (32, 32), (24, 24), (16, 16), (8, 8)]:
    t2 = np.unique(theta_tiles[:, 2])[t2_idx]
    t3 = np.unique(theta_tiles[:, 2])[t3_idx]
    selection = (theta_tiles[:, 2] == t2) & (theta_tiles[:, 3] == t3)
    Pselect = P[:, selection]
    Bselect = B[selection, :]
    utils.save_ub(
        f"P-{name}-{n_theta_1d}-{sim_size}-{t2_idx}.csv",
        f"B-{name}-{n_theta_1d}-{sim_size}-{t3_idx}.csv",
        Pselect,
        Bselect,
    )

```

```python
import numpy as np

data = np.load("berry2_64_8160.npy", allow_pickle=True)

```

```python
data = data.tolist()

```

```python
p_path = "P4D.csv"
b_path = "B4D.csv"
np.savetxt(p_path, data["P"], fmt="%s", delimiter=",")
np.savetxt(b_path, data["B"], fmt="%s", delimiter=",")

```

```python
t3 = np.unique(data["theta_tiles"][:, 2])[8]
selection = data["theta_tiles"][:, 3] == t3
p_path = "P3D.csv"
b_path = "B3D.csv"
np.savetxt(p_path, data["P"][:, selection], fmt="%s", delimiter=",")
np.savetxt(b_path, data["B"][selection, :], fmt="%s", delimiter=",")

```

```python

```
