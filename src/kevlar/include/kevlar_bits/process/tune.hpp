#pragma once
#include <ctime>
#include <limits>
#include <iostream>
#include <kevlar_bits/process/driver.hpp>
#include <kevlar_bits/util/progress_bar.hpp>

namespace kevlar {

template <class PVecType
        , class PEndptType
        , class LmdaGridType
        , class RNGGenFType
        , class ModelType
        , class PBType = pb_ostream>
auto tune(
        size_t n_sim,
        double alpha,
        double delta,
        size_t grid_dim,
        double grid_radius,
        const PVecType& p,
        const PEndptType& p_endpt,
        const LmdaGridType& lmda_grid,
        RNGGenFType rng_gen_f,
        ModelType&& model,
        size_t start_seed=time(0),
        size_t p_batch_size=std::numeric_limits<size_t>::infinity(),
        PBType&& pb = PBType(std::cout),
        bool do_progress_bar = true,
        unsigned int n_thr=std::thread::hardware_concurrency()
        )
{
    auto process_upper_bd = [](auto& upper_bd_full,
                               const auto& p_idxer_prev,
                               const auto& p,
                               const auto& p_endpt,
                               auto alpha,
                               auto thr_delta,
                               auto grid_radius,
                               auto& max_lmda_size) {
        // create the final upper bound
        upper_bd_full.create(
                p_idxer_prev, p, p_endpt,
                alpha, thr_delta, grid_radius);
        auto& upper_bd_raw = upper_bd_full.get();
        
        // check the lmda threshold condition
        for (int j = 0; j < upper_bd_raw.cols(); ++j) {
            auto col_j = upper_bd_raw.col(j);
            auto begin = col_j.data();
            auto end = begin + max_lmda_size;
            auto it = std::find_if(begin, end, [=](auto x) { return x > alpha; });
            max_lmda_size = std::distance(begin, it); 
            if (max_lmda_size <= 0) throw min_lmda_reached_error();
        }
    };
    return driver(
            n_sim, alpha, delta, grid_dim, grid_radius, p, p_endpt,
            lmda_grid, rng_gen_f, model, start_seed, p_batch_size, pb, do_progress_bar, n_thr,
            process_upper_bd);
}

} // namespace kevlar
