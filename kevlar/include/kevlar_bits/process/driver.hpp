#pragma once
#include <vector>
#include <thread>
#include <string>
#include <random>
#include <algorithm>
#include <kevlar_bits/util/d_ary_int.hpp>
#include <kevlar_bits/util/thread.hpp>
#include <kevlar_bits/util/exceptions.hpp>
#include <kevlar_bits/util/math.hpp>

namespace kevlar {

template <class PVecType
        , class PEndptType
        , class LmdaGridType
        , class RNGGenFType
        , class ModelType
        , class PBType
        , class ProcessUpperBdFType>
auto driver(
        size_t n_sim,
        double alpha,
        double delta,
        size_t grid_dim,
        size_t grid_radius,
        const PVecType& p,
        const PEndptType& p_endpt,
        const LmdaGridType& lmda_grid,
        RNGGenFType rng_gen_f,
        ModelType&& model,
        size_t start_seed,
        size_t p_batch_size,
        PBType&& pb,
        bool do_progress_bar,
        unsigned int n_thr,
        ProcessUpperBdFType process_upper_bd_f
        )
{
    using upper_bd_t = typename std::decay_t<ModelType>::upper_bd_t;

    if (n_thr <= 0) throw thread_error("Number of threads must be greater than 0.");

    // compute the normal threshold for 1-delta/2
    auto thr_delta = qnorm(1.-delta/2.);

    bool p_batch_defaulted = 
        (p_batch_size == std::numeric_limits<size_t>::infinity());

    unsigned int max_thr = std::thread::hardware_concurrency();
    n_thr = std::min(n_thr, max_thr);
    size_t n_sim_per_thr = n_sim / n_thr; 
    size_t n_sim_remain = n_sim % n_thr;

    std::vector<std::thread> pool(n_thr);
    
    // routine for each thread
    auto f = [&](int thr_i, 
                 size_t n, 
                 const dAryInt& begin, 
                 size_t p_batch_size_thr, 
                 const auto& lmda_grid,
                 upper_bd_t& upper_bd) 
    {
        // set affinity to cpu thr_i
        auto status = set_affinity(thr_i);
        if (status) {
            std::string msg = "Error calling pthread_setaffinity_np: ";
            msg += std::to_string(status);
            throw thread_error(msg);
        }

        // TODO: rng thing may need to be generalized to some class type
        // similar to how it is for upper_bd_t.
        Eigen::MatrixXd rng;
        std::mt19937 gen(start_seed + thr_i);

        // reset the upper bound object
        upper_bd.reset(lmda_grid.size(), p_batch_size_thr);

        // for each simulation, create necessary rng, run model, update upper-bound object.
        for (size_t i = 0; i < n; ++i) {
            rng_gen_f(gen, rng);
            model.run(rng, begin, p_batch_size_thr, p, lmda_grid, upper_bd);
        }

        // at this point, upper_bd is in a state where it could technically
        // compute the upper bound using the first n simulations.
    };

    dAryInt p_idxer(p.size(), grid_dim);
    dAryInt p_idxer_prev(p.size(), grid_dim);
    size_t p_grid_size = ipow(p.size(), grid_dim);
    size_t n_p_completed = 0;
    size_t max_lmda_size = lmda_grid.size();

    // construct each thread's upper bound objects.
    std::vector<upper_bd_t> upper_bds;
    upper_bds.reserve(pool.size());
    for (size_t i = 0; i < pool.size(); ++i) {
        upper_bds.emplace_back(model.make_upper_bd());
    }
    upper_bd_t upper_bd_full = model.make_upper_bd();

    // routine to run on each partition
    auto run_partition = [&](size_t p_batch_size_) {

        // MUST reset the final upper bound object
        upper_bd_full.reset(max_lmda_size, p_batch_size_);

        // run every thread
        auto lmda_subset = lmda_grid.head(max_lmda_size);
        for (size_t thr_i = 0; thr_i < pool.size()-1; ++thr_i) {
            pool[thr_i] = std::thread(f, 
                    thr_i, n_sim_per_thr, 
                    std::cref(p_idxer), p_batch_size_, 
                    lmda_subset, 
                    std::ref(upper_bds[thr_i]));
        } 
        pool.back() = std::thread(f, 
                pool.size()-1, n_sim_per_thr + n_sim_remain,
                std::cref(p_idxer), p_batch_size_, 
                lmda_subset,
                std::ref(upper_bds[pool.size()-1]));

        // increment indexer while all threads are running
        p_idxer_prev = p_idxer;
        for (size_t k = 0; k < p_batch_size_; ++k, ++p_idxer);

        // wait until all threads finish
        for (auto& thr : pool) { thr.join(); }

        // pool upper bound estimates across all threads
        for (size_t i = 0; i < upper_bds.size()-1; ++i) {
            upper_bd_full.pool(upper_bds[i], n_sim_per_thr);
        }
        upper_bd_full.pool(upper_bds.back(), n_sim_per_thr+n_sim_remain);

        process_upper_bd_f(
                upper_bd_full, p_idxer_prev, p, p_endpt,
                alpha, thr_delta, grid_radius, max_lmda_size);
    };

    // initialize progress bar object (only used if do_progress_bar is true)
    pb.set_n_total(p_grid_size);

    if (do_progress_bar) pb.initialize();

    try {
        // run the simulation on each partition of the p-grid
        while (n_p_completed < p_grid_size) {
            size_t batch_size = p_batch_size;
            if (p_batch_defaulted) {
                // get next proposed batch size
                // TODO: generalize to pass l2_cache_size_per_thr as second param!
                int proposal = upper_bd_full.p_batch_size_hint(max_lmda_size);
                batch_size = std::max(proposal, 1);
            }             
            // take the min with remaining number of coordinates
            batch_size = std::min(batch_size, p_grid_size - n_p_completed);
            run_partition(batch_size);
            n_p_completed += batch_size;
            if (do_progress_bar) pb.update(batch_size);
        }
    } 
    catch (const kevlar_error& e) {
        if (do_progress_bar) pb.finish();
        throw;
    }

    return lmda_grid[max_lmda_size-1];
}

} // namespace kevlar
