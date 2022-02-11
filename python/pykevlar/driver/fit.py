from pykevlar.core import InterSum, mt19937
import os
import copy
from multiprocessing.pool import Pool
import numpy as np

def fit_thread(model,
               sim_size,
               seed,
               thread_id=None):
    '''
    Runs simulations for a given range of grid-points and a model.

    This method uses mt19937 as the RNG.

    Parameters
    ----------
    model       :   model object.
    sim_size    :   number of simulations for each grid-point.
    seed        :   seed to random number generator.
    thread_id   :   thread ID number.

    Returns
    -------
    InterSum
    '''
    if not (thread_id is None):
        print("Enter thread {tid}".format(tid = thread_id))

    model = copy.deepcopy(model)
    model_state = model.make_state()

    is_o = InterSum(model_state.n_models(),
                    model_state.n_gridpts(),
                    model_state.n_params())

    gen = mt19937() # TODO: maybe generalize this?
    gen.seed(seed)

    for _ in range(sim_size):
        model_state.gen_rng(gen)
        model_state.gen_suff_stat()
        is_o.update(model_state)

    return is_o

def fit_process(model,
                sim_size,
                base_seed,
                n_threads=os.cpu_count()):
    '''
    Runs simulations for a given range of grid-points and a model.
    Splits the workload evenly across n_threads number of threads
    where each thread fits with sim_size /= n_threads (some threads have an additional simulation).

    NOTE: it is implementation-specific how we spawn/manage threads.

    Parameters
    ----------

    model       :   model object.
    sim_size    :   number of simulations for each grid-point.
    base_seed   :   each thread will receive a seed of base_seed + thread_id.
    n_threads   :   number of threads to spawn. Must be a positive integer. Default is os.cpu_count().

    Returns
    -------
    InterSum
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
         base_seed + i,
         i)
        for i in range(n_threads)
    ]

    with Pool(processes=n_threads) as p:
        is_os = p.starmap(fit_thread, inputs)

    # ========= END THREAD LOGIC ============

    is_final = is_os[0] # valid since len(is_os) > 0 always.
    for other in is_os[1:]:
        is_final.pool(other)

    return is_final
