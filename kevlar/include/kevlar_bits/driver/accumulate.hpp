#pragma once
#include <omp.h>

#include <cstddef>
#include <kevlar_bits/util/macros.hpp>
#include <kevlar_bits/util/types.hpp>
#include <stdexcept>
#include <thread>
#include <vector>

namespace kevlar {
namespace driver {

/*
 * Runs a sim_size number of simulations using
 * the simulation global state object, sgs,
 * which stores common data for all simulations
 * and specifies the simulation routine via the
 * simulation state class.
 * The simulations are run on the
 * grid range specified by grid_range.
 * For each simulation, the accumulator acc_o
 * accumulates information from it.
 * acc_o must be initialized properly so that
 * acc_o.pool(acc_o) and acc_o.update(...) have a well-defined behavior.
 */
template <class SGSType, class GridRangeType, class AccumType>
inline void accumulate(const SGSType& sgs, const GridRangeType& grid_range,
                       AccumType& acc_o, size_t sim_size, size_t seed,
                       size_t n_threads) {
    using acc_t = std::decay_t<AccumType>;
    using sgs_t = std::decay_t<SGSType>;
    using gen_t = typename sgs_t::gen_t;
    using uint_t = typename sgs_t::uint_t;

    size_t max_threads = std::thread::hardware_concurrency();

    if (n_threads <= 0) {
        throw std::runtime_error("n_threads must be positive.");
    }

    if (n_threads > max_threads) {
        n_threads %= max_threads;
    }

    auto sim_size_thr = sim_size / n_threads;
    auto sim_size_rem = sim_size % n_threads;

    std::vector<acc_t> acc_os(n_threads, acc_o);

    omp_set_num_threads(n_threads);

#pragma omp parallel for schedule(static)  // TODO: add some args
    for (size_t t = 0; t < n_threads; ++t) {
        auto sim_state = sgs.make_sim_state();
        gen_t gen(seed + t);
        colvec_type<uint_t> rej_len(grid_range.n_tiles());
        auto sim_size_t = sim_size_thr + (t < sim_size_rem);
        for (size_t i = 0; i < sim_size_t; ++i) {
            sim_state->simulate(gen, rej_len);
            acc_os[t].update(rej_len, *sim_state, grid_range);
        }
    }

    for (size_t j = 0; j < acc_os.size(); ++j) {
        acc_o.pool(acc_os[j]);
    }
}

}  // namespace driver
}  // namespace kevlar
