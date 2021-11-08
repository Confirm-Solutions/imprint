#pragma once
#include <ctime>
#include <limits>
#include <iostream>
#include <string_view>
#include <kevlar_bits/process/driver.hpp>
#include <kevlar_bits/util/progress_bar.hpp>
#include <kevlar_bits/util/serializer.hpp>
#include <Eigen/Core>

namespace kevlar {

template <class PVecType
        , class PEndptType
        , class RNGGenFType
        , class ModelType
        , class PBType = pb_ostream>
void fit(
        size_t n_sim,
        double alpha,
        double delta,
        size_t grid_dim,
        double grid_radius,
        const PVecType& p,
        const PEndptType& p_endpt,
        double lmda,
        RNGGenFType rng_gen_f,
        ModelType&& model,
        const std::string_view& serialize_fname,
        size_t start_seed=time(0),
        size_t p_batch_size=std::numeric_limits<size_t>::infinity(),
        PBType&& pb = PBType(std::cout),
        bool do_progress_bar = true,
        unsigned int n_thr=std::thread::hardware_concurrency()
        )
{
    Eigen::Matrix<double, 1, 1> lmda_grid(lmda); 
    Serializer s(serialize_fname.data());

    auto process_upper_bd = [&](auto& upper_bd_full,
                               const auto& p_range,
                               const auto& p_endpt,
                               auto alpha,
                               auto thr_delta,
                               auto grid_radius,
                               auto& max_lmda_size) {
        // create the final upper bound
        upper_bd_full.serialize(
                s, p_range, p_endpt,
                alpha, thr_delta, grid_radius);
    };

    driver(
        n_sim, alpha, delta, grid_dim, grid_radius, p, p_endpt,
        lmda_grid, rng_gen_f, model, start_seed, p_batch_size, pb, do_progress_bar, n_thr,
        process_upper_bd);
}

} // namespace kevlar
