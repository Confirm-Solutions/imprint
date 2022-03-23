#pragma once
#include <omp.h>

#include <cstddef>
#include <kevlar_bits/util/macros.hpp>
#include <stdexcept>
#include <thread>
#include <vector>

namespace kevlar {

template <class GenType, class ModelStateType, class InterSumType>
KEVLAR_STRONG_INLINE void fit_thread(ModelStateType& model_state,
                                     InterSumType& is_o, size_t sim_size,
                                     size_t seed) {
    GenType gen;
    gen.seed(seed);

    for (size_t i = 0; i < sim_size; ++i) {
        model_state.gen_rng(gen);
        model_state.gen_suff_stat();
        is_o.update(model_state);
    }
}

template <class GenType, class ModelType, class GridRangeType,
          class InterSumType>
inline void fit(const ModelType& model, const GridRangeType& grid_range,
                InterSumType& is_o, size_t sim_size, size_t seed,
                size_t n_threads) {
    using is_t = InterSumType;

    size_t max_threads = std::thread::hardware_concurrency();

    if (n_threads <= 0) {
        throw std::runtime_error("n_threads must be positive.");
    }

    if (n_threads > max_threads) {
        n_threads %= max_threads;
    }

    auto sim_size_thr = sim_size / n_threads;
    auto sim_size_rem = sim_size % n_threads;

    std::vector<is_t> is_os;

    for (size_t t = 0; t < n_threads; ++t) {
        is_os.emplace_back(model.n_models(), grid_range.n_tiles(),
                           grid_range.n_params());
    }

    omp_set_num_threads(n_threads);

#pragma omp parallel for schedule(static)  // TODO: add some args
    for (size_t t = 0; t < n_threads; ++t) {
        auto model_state = model.make_state();
        fit_thread<GenType>(*model_state, is_os[t],
                            sim_size_thr + (t < sim_size_rem), seed + t);
    }

    for (size_t j = 1; j < is_os.size(); ++j) {
        is_os[0].pool(is_os[j]);
    }

    std::swap(is_o, is_os[0]);
}

}  // namespace kevlar
