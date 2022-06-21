#pragma once
#include <omp.h>

#include <cstddef>
#include <imprint_bits/util/macros.hpp>
#include <imprint_bits/util/types.hpp>
#include <stdexcept>
#include <thread>
#include <vector>

namespace imprint {
namespace driver {

template <class VecSSType, class GridRangeType, class AccumType>
inline void accumulate_(const VecSSType& vec_ss,
                        const GridRangeType& grid_range, AccumType& acc_o,
                        size_t sim_size, size_t n_threads) {
    using acc_t = std::decay_t<AccumType>;
    using gr_t = GridRangeType;
    using uint_t = typename gr_t::uint_t;

    auto sim_size_thr = sim_size / n_threads;
    auto sim_size_rem = sim_size % n_threads;

    std::vector<acc_t> acc_os(n_threads, acc_o);

    assert(vec_ss.size() == n_threads);

#pragma omp parallel for schedule(static) num_threads(n_threads)
    for (size_t t = 0; t < n_threads; ++t) {
        auto& sim_state = *vec_ss[t];
        colvec_type<uint_t> rej_len(grid_range.n_tiles());
        auto sim_size_t = sim_size_thr + (t < sim_size_rem);
        for (size_t i = 0; i < sim_size_t; ++i) {
            sim_state.simulate(rej_len);
            acc_os[t].update(rej_len, sim_state, grid_range);
        }
    }

    for (size_t j = 0; j < acc_os.size(); ++j) {
        acc_o.pool(acc_os[j]);
    }
}

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
    using sgs_t = SGSType;
    using ss_t = typename sgs_t::interface_t::sim_state_t;

    size_t max_threads = std::thread::hardware_concurrency();

    if (n_threads <= 0) {
        throw std::runtime_error("n_threads must be positive.");
    }

    if (n_threads > max_threads) {
        n_threads = max_threads;
    }

    std::vector<std::unique_ptr<ss_t>> ss_s;
    ss_s.reserve(n_threads);
    for (size_t i = 0; i < n_threads; ++i) {
        ss_s.emplace_back(sgs.make_sim_state(seed + i));
    }

    accumulate_(ss_s, grid_range, acc_o, sim_size, n_threads);
}

}  // namespace driver
}  // namespace imprint
