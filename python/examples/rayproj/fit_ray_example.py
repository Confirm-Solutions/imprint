import numpy as np
import pykevlar.core as core

import ray
ray.init(address='ray://yokojitsu:10001')

print('''This cluster consists of
    {} nodes in total
    {} CPU resources in total
'''.format(len(ray.nodes()), ray.cluster_resources()['CPU']))

@ray.remote
def fit_ray_remote(input):
    import sys
    sys.path.append('/home/ray/kevlar/python')

    import pykevlar.driver as driver
    driver.fit_thread(input)
    
def fit_ray(model,
            sim_size,
            base_seed,
            n_threads):
    '''
    Runs simulations for a given range of grid-points and a model.
    Splits the workload evenly across n_threads number of threads
    where each thread fits with sim_size /= n_threads
    (some threads have an additional simulation).
    Stores the (pooled) output in a SQL database.

    NOTE: it is implementation-specific how we spawn/manage threads.
    TODO: currently we're just returning the output instead of
    storing in a database.

    Parameters
    ----------

    model       :   model object.
    sim_size    :   number of simulations for each grid-point.
    base_seed   :   each thread will receive a seed of base_seed + thread_id.
    n_threads   :   number of threads to spawn.
                    Must be a positive integer.

    Returns
    -------
    InterSum object updated with sim_size
    number of simulations under the given model.
    '''

    
    if n_threads <= 0:
        raise ValueError("n_threads must be positive.")

    sim_size_thr = sim_size // n_threads
    sim_size_rem = sim_size % n_threads

    # The following thread logic should change to something more clever.
    # ========= THREAD LOGIC ============

    # create input arguments list
    inputs = [
        (model,
         sim_size_thr + (i < sim_size_rem),
         base_seed + i)
        for i in range(n_threads)
    ]

    # with Pool(processes=n_threads) as p:
    #     is_os = p.starmap(driver.fit_thread, inputs)

    object_ids = [fit_ray_remote.remote(i) for i in inputs]
    res = ray.get(object_ids)

    # ========= END THREAD LOGIC ============

    # # Pool output from each thread (don't change!)
    # is_final = is_os[0]     # valid since len(is_os) > 0 always.
    # for other in is_os[1:]:
    #     is_final.pool(other)

    # TODO: later remove and store in database instead of returning.
    # TODO: update SQL with the rest of the upper bound quantities.
    return res

# ========== Toggleable ===============
n_arms = 3      # prioritize 3 first, then do 4
sim_size = 100000
n_thetas_1d = 64
n_threads = 2
# ========== End Toggleable ===============

ph2_size = 50
n_samples = 250
seed = 69
thresh = 1.96
lower = -0.5
upper = 0.5

# set numpy random seed
np.random.seed(seed)

# define null hypos
def null_hypo(i, p):
    return p[i] <= p[0]

# Create current batch of grid points.
# At the process-level, we only need to know theta, radii.
theta_1d = core.Gridder.make_grid(n_thetas_1d, lower, upper)
grid = np.stack(np.meshgrid(*(theta_1d for _ in range(n_arms))), axis=-1) \
        .reshape(-1, n_arms)
grid_null = np.array([
    p for p in grid if null_hypo(1, p) or null_hypo(2, p)
])
gr = core.GridRange(n_arms, grid_null.shape[0])
thetas = gr.get_thetas()
thetas[...] = np.transpose(grid_null)
radii = gr.get_radii()
radii[...] = core.Gridder.radius(n_thetas_1d, lower, upper)

# create BCKT
bckt = core.BinomialControlkTreatment(n_arms, ph2_size, n_samples, thresh)
bckt.set_grid_range(gr, null_hypo)

# run a mock-call of fit_process
is_o = fit_ray(bckt, sim_size, seed, n_threads)
print((is_o.type_I_sum() / is_o.n_accum())[0,:20])
