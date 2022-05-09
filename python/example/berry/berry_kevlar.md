---
jupyter:
  jupytext:
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

# C++ Berry-INLA version

```python
import sys
sys.path.append('../../research/berry/')
import util

import fast_inla
from scipy.special import logit, expit
import matplotlib.pyplot as plt
import numpy as np
from pykevlar import mt19937
from pykevlar.grid import HyperPlane, make_cartesian_grid_range
from pykevlar.driver import accumulate_process
import pykevlar.core.model.binomial
```

```python
fi = fast_inla.FastINLA(2)

n_arms = 2
model_type_name = 'BerryINLA' + str(n_arms)
model_type = getattr(pykevlar.core.model.binomial, model_type_name)
print(model_type)
seed = 10
n_theta_1d = 16
sim_size = 1000
n_threads = 1

# define null hypos
null_hypos = []
for i in range(n_arms):
    n = np.zeros(n_arms)
    # null is:
    # theta_i <= logit(0.1)
    # the normal should point towards the negative direction. but that also
    # means we need to negate the logit(0.1) offset
    n[i] = -1
    null_hypos.append(HyperPlane(n, -logit(0.1)))

gr = make_cartesian_grid_range(n_theta_1d, np.full(n_arms, -3.5), np.full(n_arms, 1.0), sim_size)
```

Red dots are points in the alternative hypothesis space.
Blue dots are points in the null space.

```python
gr.create_tiles(null_hypos)
plt.plot(gr.thetas()[0,:], gr.thetas()[1,:], 'ro')
gr.prune()
plt.plot(gr.thetas()[0,:], gr.thetas()[1,:], 'bo')
plt.show()
```

Run a single example set of data and make sure that the kevlar model is producing the sample results as the prototype FastINLA code.

```python
y = np.array([[4, 5]])
n = np.array([[35, 35]])
critical_values = [0.85] # final analysis exceedance requirement (note for interim analysis, the threshold)
b = model_type(
    n[0,0],
    critical_values,
    np.full(2, fi.thresh_theta),
    fi.sigma2_rule.wts.copy(),
    fi.cov.reshape((-1, 4)).T.copy(),
    fi.neg_precQ.reshape((-1, 4)).T.copy(),
    fi.logprecQdet.copy(),
    fi.log_prior.copy(),
    fi.tol,
    fi.logit_p1
)
correct = fi.numpy_inference(y, n)[1][0]
exc = b.get_posterior_exceedance_probs(y[0])
np.testing.assert_allclose(exc, correct)
```

```python
import time
start = time.time()
out = accumulate_process(b, gr, sim_size, seed, n_threads)
end = time.time()
print('runtime', end - start)
```

```python
theta = gr.thetas().T.copy()
# TODO: it'd be nice to add theta_tiles and is_null_per_arm to the GridRange object!
cum_n_tiles = np.array(gr.cum_n_tiles)
n_tiles_per_pt = cum_n_tiles[1:] - cum_n_tiles[:-1]
theta_tiles = np.repeat(theta, n_tiles_per_pt, axis=0)
```

```python
plt.figure()
plt.title('Type I error at grid points.')
plt.scatter(theta_tiles[:,0], theta_tiles[:,1], c=out.typeI_sum()[0] / sim_size)
cbar = plt.colorbar()
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
cbar.set_label('Type I error')
plt.show()
plt.title('Yellow points are above 10%')
plt.scatter(theta_tiles[:,0], theta_tiles[:,1], c=out.typeI_sum()[0] / sim_size > 0.1)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.show()
```

# Python re-implementation of accumulation


In the cell below, I generate random numbers using exactly the same mechanism as the C++ code. THIS WOULD BE UNNECESSARY in a normal situation where I'm not trying to exactly reproduce Kevlar accumulation results.

```python
%%time

# Use the mt19937 object exported from C++ so that we can match the C++ random
# sequence exactly. This is not necessary in the long term but is temporarily
# useful to ensure that this code is producing identical output to the C++
# version.
n_arm_samples = 35
gen = mt19937(seed)

# We flip the order of n_arms and n_arm_samples here so the random number
# generator produces the same sequence of uniforms as are used in the C++ kevlar
# internals. The Kevlar function operates in column-major/Fortran order. Whereas
# here, numpy operates in row-major/C ordering b
samples = np.empty((sim_size, n_arms, n_arm_samples))
gen.uniform_sample(samples.ravel())
# after transposing, samples will have shape (sim_size, n_arm_samples, n_arms)
samples = np.transpose(samples, (0, 2, 1))
```

```python
# In a normal situation, we can generate samples like this:
# np.random.seed(seed)
# samples = np.random.rand(sim_size, n_arm_samples, n_arms)
```

```python
%%time
# Calculate exceedance from each simulated sample.
# 1. Calculate the binomial count data.
# 2. See the FastINLA module for details on the inference procedure.
# 3. Success is defined by whether the exceedance probability exceeds the
#    critical value

# The sufficient statistic for binomial is just the number of uniform draws
# above the threshold probability. But the `p` array has shape (n_thetas,
# n_arms). So, we add empty dimensions to broadcast to an output `y` array of
# shape: (n_thetas, sim_size, n_arm_samples, n_arms)
theta = gr.thetas().T.copy()
# TODO: it'd be nice to add theta_tiles and is_null_per_arm to the GridRange object!
cum_n_tiles = np.array(gr.cum_n_tiles)
n_tiles_per_pt = cum_n_tiles[1:] - cum_n_tiles[:-1]
theta_tiles = np.repeat(theta, n_tiles_per_pt, axis=0)
p_tiles = expit(theta_tiles)
y = np.sum(samples[None] < p_tiles[:, None, None, :], axis=2)

# FastINLA expects inputs of shape (n, n_arms) so we must flatten our 3D arrays.
# We reshape exceedance afterwards to bring it back to 3D (n_thetas, sim_size, n_arms)
# TODO: This is where we implement the early stopping procedure.
y_flat = y.reshape((-1, 2))
n_flat = np.full_like(y_flat, n_arm_samples)
_, exceedance_flat, _, _ = fi.jax_inference(y_flat, n_flat)
exceedance = exceedance_flat.reshape(y.shape).to_py()
# instead of success, "did we reject"
success = exceedance > critical_values[0]
```

```python
%%time
# Determine type I error. 
# 1. type I is only possible when the null hypothesis is true. 
# 2. check all null hypotheses.
# 3. sum across all the simulations.
is_null_per_arm = np.array([[gr.check_null(i, j) for j in range(n_arms)] for i in range(p_tiles.shape[0])])
false_reject = success & is_null_per_arm[:, None,]
any_rejection = np.any(false_reject, axis=-1)
typeI_sum = any_rejection.sum(axis=-1)
```

```python
%%time
# The score function is the primary component of the typeI gradient:
# 1. for binomial, it's just: y - n * p
# 2. only summed when there is a rejection in the given simulation
score = y - n_arm_samples * p_tiles[:, None, :]
typeI_score = np.sum(any_rejection[:, :, None] * score, axis=1)
```

Confirm that Kevlar and this Python code produce the same output.

```python
typeI_good = np.all(out.typeI_sum() == typeI_sum)
score_good = np.all(np.abs(out.score_sum().reshape((-1, 2)) - typeI_score) < 1e-13)
typeI_good, score_good
```

```python
# y_avg = np.mean(y, axis=1)
# plt.scatter(theta_tiles[:,0], theta_tiles[:,1], c=y_avg[:,0] / n_arm_samples)
# plt.colorbar()
# plt.show()
# plt.title()
# plt.scatter(theta_tiles[:,0], theta_tiles[:,1], c=y_avg[:,1] / n_arm_samples)
# plt.colorbar()
# plt.show()
pos_start = gr.cum_n_tiles[:-1]
is_null_per_arm_gridpt = np.add.reduceat(is_null_per_arm, pos_start, axis=0) > 0

plt.title('Is null for arm 0?')
plt.scatter(theta[:,0], theta[:,1], c=is_null_per_arm_gridpt[:,0], cmap='Set1')
plt.hlines(logit(0.1), -4, 2)
plt.vlines(logit(0.1), -4, 2)
plt.xlim(np.min(theta[:,0]) - 0.2, np.max(theta[:,0]) + 0.2)
plt.ylim(np.min(theta[:,1]) - 0.2, np.max(theta[:,1]) + 0.2)
plt.colorbar()
plt.show()
plt.title('Is null for arm 1?')
plt.scatter(theta[:,0], theta[:,1], c=is_null_per_arm_gridpt[:,1], cmap='Set1')
plt.hlines(logit(0.1), -4, 2)
plt.vlines(logit(0.1), -4, 2)
plt.xlim(np.min(theta[:,0]) - 0.2, np.max(theta[:,0]) + 0.2)
plt.ylim(np.min(theta[:,1]) - 0.2, np.max(theta[:,1]) + 0.2)
plt.colorbar()
plt.show()

plt.title('type I error fraction of trials')
plt.scatter(theta_tiles[:,0], theta_tiles[:,1], c=typeI_sum / sim_size)
plt.colorbar()
plt.show()

plt.title('Tile count per grid point')
plt.scatter(theta[:,0], theta[:,1], c=n_tiles_per_pt)
plt.colorbar()
plt.show()
```

# Comparing against MCMC for a few grid points

This is a check to determine how much of the type I error is real versus an artifact coming from INLA.



```python
import mcmc
```

```python
sorted_idxs = np.argsort(typeI_sum)
idx = sorted_idxs[-1]
p_tiles[idx]
```

```python
y_mcmc = y[idx, :]
n_mcmc = np.full((sim_size, n_arms), n_arm_samples)
data_mcmc = np.stack((y_mcmc, n_mcmc), axis=-1)
n_mcmc_sims = 1000
results_mcmc = mcmc.mcmc_berry(
    data_mcmc[:n_mcmc_sims], fi.logit_p1, np.full(n_mcmc_sims, fi.thresh_theta), n_arms=2
)
success_mcmc = results_mcmc["exceedance"] > critical_values[0]
```

```python
import pickle
with open(f'berry_kevlar_mcmc{idx}.pkl', 'wb') as f:
    pickle.dump(results_mcmc, f)
```

```python
mcmc_typeI = np.sum(np.any(success_mcmc & is_null_per_arm[idx, None,:], axis=-1), axis=-1)
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
bad_sim_idxs = np.where(np.any((success[idx] & (success_mcmc)) & is_null_per_arm[idx, None], axis=-1))[0]
print('theta: ', theta_tiles[idx])
print('unique y where INLA says type 1 and MCMC also says type 1: ', np.unique(y[idx, bad_sim_idxs], axis=0))
```

```python

bad_sim_idxs = np.where(np.any((~success[idx] & (~success_mcmc)) & is_null_per_arm[idx, None], axis=-1))[0]
print('theta: ', theta_tiles[idx])
print('unique y where INLA says type 1 and MCMC also says type 1: ', np.unique(y[idx, bad_sim_idxs], axis=0))
```

# Building a rejection table

```python
ys = np.arange(n_arm_samples + 1)
Ygrids = np.stack(np.meshgrid(*[ys] * fi.n_arms, indexing='ij'), axis=-1)
Ygrids.shape
```

```python
Yravel = Ygrids.reshape((-1, fi.n_arms))
Yravel.shape
```

```python
is_sorted = np.logical_and.reduce([Yravel[:, i + 1] >= Yravel[:, i] for i in range(fi.n_arms - 1)])
is_sorted.sum() / Yravel.shape[0]
```

```python
Y_sorted = Yravel[is_sorted]
```

```python
colsortidx = np.argsort(Yravel, axis=-1)
inverse_colsortidx = np.zeros(Yravel.shape, dtype=np.int32)
inverse_colsortidx[np.arange(Yravel.shape[0])[:, None], colsortidx] = np.arange(fi.n_arms)
```

```python

Y_colsorted = Yravel[np.arange(Yravel.shape[0])[:,None],colsortidx]
Y_colsorted[-10:]
```

```python
np.all(Y_colsorted[np.arange(Yravel.shape[0])[:,None],inverse_colsortidx] == Yravel)
```

```python
Y_unique, inverse_unique = np.unique(Y_colsorted, axis=0, return_inverse=True)
Y_unique.shape
```

```python
np.all(Y_colsorted == Y_unique[inverse_unique])
```

```python
a = 0
for i in range(36):
    for j in range(i, 36):
        for k in range(j, 36):
            for q in range(k, 36):
                a += 1
a
```

```python
%%time
N = np.full_like(Y_unique, n_arm_samples)
reject_unique = fi.rejection_inference(Y_unique, N, method='jax')
reject = reject_unique[inverse_unique][np.arange(Yravel.shape[0])[:, None], inverse_colsortidx]
```

```python
n_test = 100
rand_idxs = np.random.randint(0, reject.shape[0], size=(n_test,))
np.all(reject[rand_idxs] == fi.rejection_inference(Yravel[rand_idxs], np.full((n_test, 4), n_arm_samples)))
```

```python
import jax.numpy as jnp
import jax
@jax.jit
def rejection_table(y, n):
    y_index = (y * (36 ** jnp.arange(4)[::-1])[None,:]).sum(axis=-1)
    return reject[y_index, :]
```

```python
y = np.random.randint(0, n_arm_samples + 1, size=(n_test,4))
np.all(rejection_table(y, None) == fi.rejection_inference(y, np.full((n_test, 4), n_arm_samples)))
```

```python
theta_tiles = grid.theta_tiles(gr)
is_null_per_arm = grid.is_null_per_arm(gr)
```

```python
reject.shape
```

```python
def fnc(x):
    return x + 1
```

```python
import multiprocessing as mp
mp_start_count = 0

if __name__ == '__main__':
    if mp_start_count == 0:
        try:
            mp.set_start_method('fork')
        except RuntimeError:
            # This error happens when the cell is run a second time because the
            # start method cannot be changed after the multiprocessing context
            # has already been set.
            pass
        mp_start_count += 1
    p = mp.Pool(2)
    print(p.map(fnc, range(2)))
```

```python
def chunk_accum(seed, chunk_size):
    np.random.seed(seed)
    typeI_sum = np.zeros(theta_tiles.shape[0])
    typeI_score = np.zeros(theta_tiles.shape)
    accumulator = binomial_accumulator(rejection_table)
    for i in range(1):
        samples = np.random.uniform(0, 1, size=(1, n_arm_samples, fi.n_arms))
        chunk_typeI_sum, chunk_typeI_score = accumulator(theta_tiles, is_null_per_arm, samples)
        typeI_sum += chunk_typeI_sum
        typeI_score += chunk_typeI_score
    return typeI_sum, typeI_score
```

```python
n_cores = 1
# Split sim_size
chunk_size = int(np.floor(sim_size / n_cores))
chunk_sizes = np.full(n_cores, chunk_size)
# Spread the remainder over the chunks.
for i in range(sim_size - (chunk_size * n_cores)):
    chunk_sizes[i] += 1
print(chunk_sizes)
import multiprocessing
p = multiprocessing.Pool(n_cores)
seeds = 10 + np.arange(n_cores)
p.starmap(chunk_accum, zip(seeds, chunk_sizes))
```

```python
%%time
np.random.seed(10)
typeI_sum = np.zeros(theta_tiles.shape[0])
typeI_score = np.zeros(theta_tiles.shape)
accumulator = binomial_accumulator(rejection_table)
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
slice2 = (-1.2, -1)
slice3 = (-1.2, -1)
# slice2 = (-3, -2.5)
# slice3 = (-3, -2.5)
selection = (
    (theta_tiles[:,2] > slice2[0]) & (theta_tiles[:,2] < slice2[1])
    & (theta_tiles[:,3] > slice3[0]) & (theta_tiles[:,3] < slice3[1])
)
plt.figure(figsize=(8,8))
plt.scatter(theta_tiles[selection,0], theta_tiles[selection,1], c=typeI_sum[selection]/sim_size)
plt.colorbar()
plt.show()
```

# Running a 4D grid

```python
import os

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
from pykevlar import mt19937
import pykevlar.grid as grid
from pykevlar.driver import accumulate_process
import pykevlar.core.model.binomial
import binomial

import util
import fast_inla

name = 'berry1'
fi = fast_inla.FastINLA(4)
seed = 10
n_arm_samples = 35
n_theta_1d = 32
sim_size = 10000


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

```

```python
loaded = np.load(f"{name}_{n_theta_1d}_{sim_size}.npy", allow_pickle=True).tolist()
typeI_sum = loaded["sum"]
typeI_score = loaded["score"]
table = loaded["table"]
theta_tiles = loaded["theta_tiles"]
is_null_per_arm = loaded["is_null_per_arm"]
```

```python
%%time
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

# Playing with parallelizing JAX

```python
import time
import jax
import jax.numpy as jnp

reject_fnc = lambda y, n: binomial.lookup_rejection(table, y, n)
accum = binomial.binomial_accumulator(reject_fnc)
paccum = jax.pmap(accum, in_axes=(None, None, 0), out_axes=(0, 0), axis_name='i')
# paccum = jax.pmap(accum, in_axes=(None, None, 0), out_axes=(None, None), axis_name='i')
```

```python
%%time
np.random.seed(seed)
typeI_sum = np.zeros(theta_tiles.shape[0])
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
np.save(
    f"{name}_{n_theta_1d}_{sim_size}.npy",
    dict(
        sum=typeI_sum,
        score=typeI_score,
        table=table,
        theta_tiles=theta_tiles,
        is_null_per_arm=is_null_per_arm,
    ),
    allow_pickle=True,
)

```

# A multiprocessing parallel version.

```python
import time
def chunk_accum(seed, chunk_size):
    start = time.time()
    reject_fnc = lambda y, n: binomial.lookup_rejection(table, y, n)
    np.random.seed(seed)
    typeI_sum = np.zeros(theta_tiles.shape[0])
    typeI_score = np.zeros(theta_tiles.shape)
    print(time.time() - start)
    start = time.time()
    accumulator = binomial.binomial_accumulator(reject_fnc)
    print(time.time() - start)
    start = time.time()
    for i in range(chunk_size):
        samples = np.random.uniform(0, 1, size=(1, n_arm_samples, fi.n_arms))
        chunk_typeI_sum, chunk_typeI_score = accumulator(theta_tiles, is_null_per_arm, samples)
        typeI_sum += chunk_typeI_sum
        typeI_score += chunk_typeI_score
    print(time.time() - start)
    start = time.time()
    return typeI_sum, typeI_score
```

```python
import cloudpickle
```

```python
%%time
n_cores = 8
# Split sim_size
chunk_size = int(np.floor(sim_size / n_cores))
chunk_sizes = np.full(n_cores, chunk_size)
# Spread the remainder over the chunks.
for i in range(sim_size - (chunk_size * n_cores)):
    chunk_sizes[i] += 1
print(chunk_sizes)
import multiprocessing
p = multiprocessing.Pool(n_cores)
seeds = 10 + np.arange(n_cores)
fnc_pkl = [cloudpickle.dumps(chunk_accum)] * len(seeds)
args = list(zip(fnc_pkl, seeds, chunk_sizes))
```

```python
%%time
results = p.starmap(binomial.cloudpickle_helper, args)
# results = []
# for i in range(n_cores):
#     print(i)
#     results.append(chunk_accum(*list(args)[i][1:]))
```

# Making some figures!

```python
typeI_sum.shape
```

```python
%config InlineBackend.figure_format='retina'
factor = 0.75
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["font.size"] = 20 * factor
plt.rcParams["axes.labelsize"] = 18 * factor
plt.rcParams["axes.titlesize"] = 20 * factor
plt.rcParams["xtick.labelsize"] = 16 * factor
plt.rcParams["ytick.labelsize"] = 16 * factor
plt.rcParams["legend.fontsize"] = 20 * factor
plt.rcParams["figure.titlesize"] = 22 * factor
plt.rcParams["axes.facecolor"] = (1.0, 1.0, 1.0, 1.0)
plt.rcParams["figure.facecolor"] = (1.0, 1.0, 1.0, 1.0)
plt.rcParams["savefig.transparent"] = False
plt.rcParams["image.cmap"] = "plasma"
```

```python
# slice2 = (-1.2, -1.1)
# slice3 = (-1.2, -1.1)
for t2_idx, t3_idx in [(16, 16), (8, 8)]:
    t2 = np.unique(theta_tiles[:, 2])[t2_idx]
    t3 = np.unique(theta_tiles[:, 2])[t3_idx]
    selection = (theta_tiles[:,2] == t2) & (theta_tiles[:,3] == t3)

    plt.figure(figsize=(6,6), constrained_layout=True)
    plt.title(f'slice: ($\\theta_2, \\theta_3) \\approx$ ({t2:.1f}, {t3:.1f})')
    plt.scatter(theta_tiles[selection,0], theta_tiles[selection,1], c=typeI_sum[selection]/sim_size, s=120)
    cbar = plt.colorbar()
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    cbar.set_label('Fraction of Type I errors')
    plt.savefig(f'grid_type1_{t2_idx}_{t3_idx}.png', dpi=300, bbox_inches='tight')
    plt.show()
```

```python
np.sum(selection)
```

```python
np.unique(theta_tiles[:,2])
```

```python

```
