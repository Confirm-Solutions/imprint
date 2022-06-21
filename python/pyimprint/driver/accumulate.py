import os

from pyimprint.core.bound import TypeIErrorAccum
from pyimprint.core.driver import accumulate


def accumulate_process(
    model, grid_range, sim_size, base_seed, n_threads=os.cpu_count()
):
    """
    Runs simulations for a given range of grid-points and a model.
    Splits the workload evenly across n_threads number of threads
    where each thread accumulates with sim_size /= n_threads
    (some threads have an additional simulation).
    Stores the (pooled) output in a SQL database.

    NOTE: it is implementation-specific how we spawn/manage threads.
    TODO: currently we're just returning the output instead of
    storing in a database.

    Parameters
    ----------

    model       :   model object.
    grid_range  :   grid range object.
    sim_size    :   number of simulations for each grid-point.
    base_seed   :   each thread will receive a seed of base_seed + thread_id.
    n_threads   :   number of threads to spawn.
                    Must be a positive integer.
                    Default is os.cpu_count().

    Returns
    -------
    InterSum object updated with sim_size
    number of simulations under the given model.
    """

    if n_threads <= 0:
        raise ValueError("n_threads must be positive.")

    max_threads = os.cpu_count()
    if n_threads > max_threads:
        n_threads = n_threads % max_threads

    # create sim global state
    sgs = model.make_sim_global_state(grid_range)

    # create sim states
    ss_s = [sgs.make_sim_state(base_seed + i) for i in range(n_threads)]

    # prepare output
    acc_o = TypeIErrorAccum(
        model.n_models(), grid_range.n_tiles(), grid_range.n_params()
    )

    # run C++ core routine
    accumulate(
        vec_sim_states=ss_s,
        grid_range=grid_range,
        accum=acc_o,
        sim_size=sim_size,
        n_threads=n_threads,
    )

    return acc_o


def accumulate_driver(batcher, model, base_seed, n_threads=os.cpu_count()):
    """
    Batches grid points using batcher
    and simulates each batch on a node in a cluster.

    TODO: for PoC, we're currently just sequentially
    processing each batch and yielding each result.
    Eventually, once accumulate_process stores into SQL,
    accumulate_driver doesn't need to yield or output anything.

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
    """

    for batch, sim_size in batcher:
        # TODO: accumulate_process won't output anything later
        acc_o = accumulate_process(
            model=model,
            grid_range=batch,
            sim_size=sim_size,
            base_seed=base_seed,
            n_threads=n_threads,
        )
        # TODO: no need to yield anything later
        yield acc_o
