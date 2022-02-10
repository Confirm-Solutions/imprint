# TODO: need access to pybind exported InterSum

def fit_thread(model_state,
               sim_size,
               gen):
    '''
    Runs simulations for a given range of grid-points under a given model
    using a seed for RNG.

    Parameters
    ----------
    model_state :   model state object.
    sim_size    :   number of simulations for each grid-point.
    gen         :   random number generator.

    Returns
    -------
    InterSum
    '''

    is_o = InterSum(model_state.n_models(),
                    model_state.n_gridpts(),
                    model_state.n_params())

    for _ in range(sim_size):
        model_state.get_rng(gen)
        model_state.get_suff_stat()
        is_o.update(model_state)

    return is_o
