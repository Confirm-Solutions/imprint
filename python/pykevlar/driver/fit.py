from pykevlar.core import InterSum, mt19937
import os
from multiprocessing.pool import Pool
# TODO: temporary
from timeit import default_timer as timer
from datetime import timedelta


def fit_thread(model,
               grid_range,
               sim_size,
               seed):
    '''
    Runs simulations for a given range of grid-points and a model.
    Returns the updated InterSum object.

    This method uses mt19937 as the RNG.

    Parameters
    ----------
    model           :   model object.
    sim_size        :   number of simulations for each grid-point.
    seed            :   seed to random number generator.

    Returns
    -------
    InterSum object updated with sim_size
    number of simulations under the given model.
    '''

    model.set_grid_range(grid_range)
    model_state = model.make_state()

    is_o = InterSum(model.n_models(),
                    grid_range.n_tiles(),
                    grid_range.n_params())

    gen = mt19937()     # TODO: maybe generalize this?
    gen.seed(seed)

    start = timer()
    for _ in range(sim_size):
        model_state.gen_rng(gen)
        model_state.gen_suff_stat()
        is_o.update(model_state)
    end = timer()
    print("Thread time: {t}".format(t=timedelta(seconds=end-start)))

    return is_o


def fit_process(model,
                grid_range,
                sim_size,
                base_seed,
                n_threads=os.cpu_count()):
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
                    Default is os.cpu_count().

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
         grid_range,
         sim_size_thr + (i < sim_size_rem),
         base_seed + i)
        for i in range(n_threads)
    ]

    with Pool(processes=n_threads) as p:
        is_os = p.starmap(fit_thread, inputs)

    # ========= END THREAD LOGIC ============

    # Pool output from each thread (don't change!)
    is_final = is_os[0]     # valid since len(is_os) > 0 always.
    for other in is_os[1:]:
        is_final.pool(other)

    # TODO: later remove and store in database instead of returning.
    # TODO: update SQL with the rest of the upper bound quantities.
    return is_final


def fit_driver(batcher,
               model,
               base_seed,
               n_threads=os.cpu_count()):
    '''
    Batches grid points using batcher
    and simulates each batch on a node in a cluster.

    TODO: for PoC, we're currently just sequentially
    processing each batch and yielding each result.
    Eventually, once fit_process stores into SQL,
    fit_driver doesn't need to yield or output anything.

    Parameters
    ----------

    batcher     :   object that batches grid points.
                    Must be iterable where each iterator
                    returns the next batch of grid points
                    as a GridRange and the number
                    of simulation size to run.
    model       :   model object.
    base_seed   :   base seed for each node (process).
    n_threads   :   number of threads to spawn in each node.

    Returns
    -------

    Yields each InterSum output for each batch
    '''

    for batch, sim_size in batcher:
        # TODO: fit_process won't output anything later
        is_o = fit_process(model=model,
                           grid_range=batch,
                           sim_size=sim_size,
                           base_seed=base_seed,
                           n_threads=n_threads)
        # TODO: no need to yield anything later
        yield is_o
