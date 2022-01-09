#pragma once
#include <vector>
#include <ctime>
#include <thread>
#include <string>
#include <random>
#include <algorithm>
#include <limits>
#include <iostream>
#include <string_view>
#include <kevlar_bits/util/d_ary_int.hpp>
#include <kevlar_bits/util/thread.hpp>
#include <kevlar_bits/util/exceptions.hpp>
#include <kevlar_bits/util/math.hpp>
#include <kevlar_bits/util/range/grid_range.hpp>
#include <kevlar_bits/util/progress_bar.hpp>
#include <kevlar_bits/util/serializer.hpp>
#include <kevlar_bits/model/upper_bound.hpp>
#include <Eigen/Core>

namespace kevlar {
namespace internal {

/* Forward Declaration */
template <class T> struct traits;

/*
 * Routine run by each thread.
 * This routine initializes RNG using start_seed as the base seed.
 * It then simulates n times and accumulates any necessary quantities into upper_bound.
 * It assumes upper_bound was properly initialized 
 * (see the code below to see how upper_bound interacts with state).
 *
 * @param   thr_i               thread ID (usually 0, 1, ..., n_cpus).
 * @param   n                   simulation size.
 * @param   start_seed          seed base. The actual seed used is start_seed + thr_i.
 * @param   mean_idxer_range    a range object defining a range of grid points.
 * @param   lmda_grid           critical thresholds to optimize over. 
 *                              Assumes it is monotonic in the correct direction,
 *                              e.g. if the underlying test is 1-sided (upper),
 *                              lmda_grid should be a decreasing set of thresholds,
 *                              so that the test goes from most conservative to least conservative.
 * @param   state               a state object to generate a simulation and sufficient statistics.
 *                              See upper_bound.hpp to see how it interfaces with the state object.
 *                              See binomial_control_k_treatment.hpp to see an example of a state class.
 * @param   upper_bound         upper bound object to accumulate each simulation result.
 */
template <class MeanIdxerRangeType
        , class LmdaType
        , class StateType
        , class UpperBoundType>
void thread_routine(
    int thr_i, 
    size_t n, 
    size_t start_seed,
    const MeanIdxerRangeType& mean_idxer_range,
    const LmdaType& lmda_grid,
    StateType& state,
    UpperBoundType& upper_bd
        )
{
    // set affinity to cpu thr_i
    auto status = set_affinity(thr_i);
    if (status) {
        std::string msg = "Error calling pthread_setaffinity_np: ";
        msg += std::to_string(status);
        throw thread_error(msg);
    }

    // initialize RNG
    std::mt19937 gen(start_seed + thr_i);

    // for each simulation, create necessary rng, sufficient statistics, and update upper-bound object.
    for (size_t i = 0; i < n; ++i) {
        state.gen_rng(gen);
        state.gen_suff_stat();
        upper_bd.update(state, mean_idxer_range, lmda_grid);
    }

    // at this point, upper_bd is in a state where it could 
    // compute the upper bound using these n simulations.
}

/*
 * This function is the main driver for running all models.
 * @param   model               model to run.
 * @param   n_sim               number of total simulations.
 * @param   delta               confidence for provable upper bound.
 * @param   grid_radius         radius along each direction in Theta space.
 *                              Note: for now, we assume this radius is the same across all directions.
 *                              The gridding of Theta space is based on this radius.
 * @param   lmda_grid           critical thresholds to optimize over. 
 *                              Assumes it is monotonic in the correct direction,
 *                              e.g. if the underlying test is 1-sided (upper),
 *                              lmda_grid should be a decreasing set of thresholds,
 *                              so that the test goes from most conservative to least conservative.
 * @param   start_seed          base start seed. 
 * @param   p_batch_size        batch size for grid points. If p_batch_size == std::numeric_limits<double>::infinity(),
 *                              we default to our own batching scheme.
 * @param   pb                  progress bar object.
 * @param   do_progress_bar     updates progress bar if true.
 * @param   n_thr               number of threads (will be modified to be capped at number of CPUs).
 * @param   process_upper_bound_f   functor to process the final upper bound object at every batch.
 */
template <class ModelType
        , class LmdaGridType
        , class PBType
        , class ProcessUpperBdFType>
auto driver(
        ModelType&& model,
        size_t n_sim,
        double delta,
        double grid_radius,
        const LmdaGridType& lmda_grid,
        size_t start_seed,
        size_t p_batch_size,
        PBType&& pb,
        bool do_progress_bar,
        unsigned int n_thr,
        ProcessUpperBdFType process_upper_bd_f
        )
{
    using model_t = std::decay_t<ModelType>;
    using state_t = typename internal::traits<model_t>::state_t;
    using upper_bd_t = UpperBound<double>;

    if (n_thr <= 0) throw thread_error("Number of threads must be greater than 0.");

    bool p_batch_defaulted = 
        (p_batch_size == std::numeric_limits<size_t>::infinity());

    unsigned int max_thr = std::thread::hardware_concurrency();
    n_thr = std::min(n_thr, max_thr);
    size_t n_sim_per_thr = n_sim / n_thr; 
    size_t n_sim_remain = n_sim % n_thr;

    std::vector<std::thread> pool(n_thr);

    rectangular_range curr_range(model.n_means(), model.n_arms(), 0);
    rectangular_range prev_range = curr_range;
    size_t mean_grid_size = ipow(model.n_means(), model.n_arms());
    size_t n_p_completed = 0;
    size_t max_lmda_size = lmda_grid.size();

    std::vector<state_t> states;
    states.reserve(pool.size());
    for (size_t i = 0; i < pool.size(); ++i) {
        states.emplace_back(model);
    }

    // construct each thread's upper bound objects.
    std::vector<upper_bd_t> upper_bds(pool.size());

    // construct the final upper bound object that will pool from upper_bds.
    upper_bd_t upper_bd_full;

    // routine to run on each partition
    auto run_partition = [&](size_t p_batch_size_) {

        // MUST reset the final upper bound object
        // and reset the range object.
        upper_bd_full.reset(max_lmda_size, p_batch_size_, model.n_arms());
        curr_range.set_size(p_batch_size_);

        // run every thread
        auto lmda_subset = lmda_grid.head(max_lmda_size);
        auto thr_routine_wrapper = 
            [](int thr_i,
               size_t n_sim_per_thr,
               size_t start_seed,
               const rectangular_range& curr_range,
               const auto& lmda_subset,
               state_t& state,
               upper_bd_t& upper_bd) 
            { 
                thread_routine(
                        thr_i, n_sim_per_thr, start_seed, 
                        curr_range, lmda_subset, 
                        state, upper_bd);
            };

        for (size_t thr_i = 0; thr_i < pool.size(); ++thr_i) {
            // the last thread will take the remaining simulations also.
            size_t n_sim_thr = (thr_i != pool.size()-1) ? 
                n_sim_per_thr : (n_sim_per_thr + n_sim_remain);
            // reset the upper bound object
            upper_bds[thr_i].reset(lmda_subset.size(), curr_range.size(), model.n_arms());
            // run each thread
            pool[thr_i] = std::thread(
                    thr_routine_wrapper, 
                    thr_i, n_sim_thr, start_seed, 
                    curr_range,
                    lmda_subset, std::ref(states[thr_i]),
                    std::ref(upper_bds[thr_i]) );
        } 

        // Increment indexer while all threads are running. 
        // Pre-emptive update for some optimization.
        prev_range = curr_range;
        auto& curr_idxer = curr_range.get_idxer();
        for (size_t k = 0; k < p_batch_size_; ++k, ++curr_idxer);

        // wait until all threads finish
        for (auto& thr : pool) { thr.join(); }

        // pool upper bound estimates across all threads
        for (size_t i = 0; i < upper_bds.size(); ++i) {
            upper_bd_full.pool(upper_bds[i]);
        }

        process_upper_bd_f(
                upper_bd_full, prev_range,
                delta, grid_radius, max_lmda_size);
    };

    // initialize progress bar object 
    if (do_progress_bar) {
        pb.set_n_total(mean_grid_size);
        pb.initialize();
    }

    try {
        // run the simulation on each partition of the p-grid
        while (n_p_completed < mean_grid_size) {
            size_t batch_size = p_batch_size;
            if (p_batch_defaulted) {
                // get next proposed batch size
                int proposal = upper_bd_full.p_batch_size_hint(max_lmda_size);
                batch_size = std::max(proposal, 1);
            }             
            // take the min with remaining number of grid points.
            batch_size = std::min(batch_size, mean_grid_size - n_p_completed);
            run_partition(batch_size);
            n_p_completed += batch_size;
            if (do_progress_bar) pb.update(batch_size);
        }
    } 
    catch (const kevlar_error& e) {
        if (do_progress_bar) pb.finish();
        throw;
    }

    // output the least conservative threshold that makes the upper bound <= alpha everywhere.
    return lmda_grid[max_lmda_size-1];
}

} // namespace internal

/*
 * Fits a model (design) using n_sim simulations with confidence 1-delta.
 * The grid is assumed to have grid_radius in all directions.
 * The test will reject with threshold lmda.
 * The outputted upper bound object will be stored in serialize_fname.
 *
 * @param   model               model to run.
 * @param   n_sim               number of total simulations.
 * @param   delta               confidence for provable upper bound.
 * @param   grid_radius         radius along each direction in Theta space.
 *                              Note: for now, we assume this radius is the same across all directions.
 *                              The gridding of Theta space is based on this radius.
 * @param   lmda                critical threshold. 
 * @param   serialize_fname     file name to output the serialized upper bound output.
 * @param   start_seed          base start seed.
 * @param   p_batch_size        batch size for grid points. If p_batch_size == std::numeric_limits<double>::infinity(),
 *                              we default to our own batching scheme.
 * @param   pb                  progress bar object.
 * @param   do_progress_bar     updates progress bar if true.
 * @param   n_thr               number of threads (will be modified to be capped at number of CPUs).
 */
template <class ModelType, class PBType = pb_ostream>
void fit(
        ModelType&& model,
        size_t n_sim,
        double delta,
        double grid_radius,
        double lmda,
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
                               auto delta,
                               auto grid_radius,
                               auto& max_lmda_size) {
        // serialize the final upper bound
        upper_bd_full.serialize(s, model, p_range, delta, grid_radius);
    };

    internal::driver(
        model, n_sim, delta, grid_radius, 
        lmda_grid, start_seed, p_batch_size, pb, do_progress_bar, n_thr,
        process_upper_bd);
}

/*
 * Tunes a model (design) using n_sim simulations with confidence 1-delta.
 * The grid is assumed to have grid_radius in all directions.
 * The test will reject with threshold lmda in lmda_grid.
 * The outputted lmda is the smallest threshold (least conservative) such that
 * upper bound values at all grid points are <= alpha.
 *
 * @param   model               model to run.
 * @param   n_sim               number of total simulations.
 * @param   alpha               Type I error nominal level.
 * @param   delta               confidence for provable upper bound.
 * @param   grid_radius         radius along each direction in Theta space.
 *                              Note: for now, we assume this radius is the same across all directions.
 *                              The gridding of Theta space is based on this radius.
 * @param   lmda_grid           critical thresholds to optimize over. 
 *                              Assumes it is monotonic in the correct direction,
 *                              e.g. if the underlying test is 1-sided (upper),
 *                              lmda_grid should be a decreasing set of thresholds,
 *                              so that the test goes from most conservative to least conservative.
 * @param   start_seed          base start seed.
 * @param   p_batch_size        batch size for grid points. If p_batch_size == std::numeric_limits<double>::infinity(),
 *                              we default to our own batching scheme.
 * @param   pb                  progress bar object.
 * @param   do_progress_bar     updates progress bar if true.
 * @param   n_thr               number of threads (will be modified to be capped at number of CPUs).
 */
template <class ModelType
        , class LmdaGridType
        , class PBType = pb_ostream>
auto tune(
        ModelType&& model,
        size_t n_sim,
        double alpha,
        double delta,
        double grid_radius,
        const LmdaGridType& lmda_grid,
        size_t start_seed=time(0),
        size_t p_batch_size=std::numeric_limits<size_t>::infinity(),
        PBType&& pb = PBType(std::cout),
        bool do_progress_bar = true,
        unsigned int n_thr=std::thread::hardware_concurrency()
        )
{
    auto process_upper_bd = [&](auto& upper_bd_full,
                                const auto& p_range,
                                auto delta,
                                auto grid_radius,
                                auto& max_lmda_size) {
        // create the final upper bound
        upper_bd_full.create(model, p_range, delta, grid_radius);
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
    return internal::driver(
            model, n_sim, delta, grid_radius, 
            lmda_grid, start_seed, p_batch_size, pb, do_progress_bar, n_thr,
            process_upper_bd);
}

} // namespace kevlar
