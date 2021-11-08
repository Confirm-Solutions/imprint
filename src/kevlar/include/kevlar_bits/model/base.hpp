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
#include <Eigen/Core>

namespace kevlar {
namespace internal {

template <class T>
struct traits;

} // namespace internal

template <class Derived>
struct ModelBase
{
private:
    using derived_t = Derived;
    derived_t& self() { return static_cast<derived_t&>(*this); }
    const derived_t& self() const { return static_cast<const derived_t&>(*this); }

    template <class PRangeType
            , class LmdaType
            , class RngGenFType
            , class UpperBoundType>
    void thread_routine(
        int thr_i, 
        size_t n, 
        size_t start_seed,
        const PRangeType& p_range,
        const LmdaType& lmda_grid,
        RngGenFType rng_gen_f,
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

        // TODO: rng thing may need to be generalized to some class type
        // similar to how it is for upper_bd_t.
        Eigen::MatrixXd rng;
        std::mt19937 gen(start_seed + thr_i);

        // reset the upper bound object
        upper_bd.reset(lmda_grid.size(), p_range.size());

        // for each simulation, create necessary rng, run model, update upper-bound object.
        for (size_t i = 0; i < n; ++i) {
            rng_gen_f(gen, rng);
            self().run(rng, p_range, lmda_grid, upper_bd);
        }

        // at this point, upper_bd is in a state where it could technically
        // compute the upper bound using the first n simulations.
    }

public:
    template <class PVecType
            , class PEndptType
            , class LmdaGridType
            , class RNGGenFType
            , class PBType
            , class ProcessUpperBdFType>
    auto driver(
            size_t n_sim,
            double alpha,
            double delta,
            size_t grid_dim,
            double grid_radius,
            const PVecType& p,
            const PEndptType& p_endpt,
            const LmdaGridType& lmda_grid,
            RNGGenFType rng_gen_f,
            size_t start_seed,
            size_t p_batch_size,
            PBType&& pb,
            bool do_progress_bar,
            unsigned int n_thr,
            ProcessUpperBdFType process_upper_bd_f
            )
    {
        using upper_bd_t = typename internal::traits<derived_t>::upper_bd_t;

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

        using rectangular_range_t = rectangular_range<std::decay_t<PVecType> >;
        rectangular_range curr_range(p, grid_dim, 0);
        rectangular_range prev_range = curr_range;
        size_t p_grid_size = ipow(p.size(), grid_dim);
        size_t n_p_completed = 0;
        size_t max_lmda_size = lmda_grid.size();

        // construct each thread's upper bound objects.
        std::vector<upper_bd_t> upper_bds;
        upper_bds.reserve(pool.size());
        for (size_t i = 0; i < pool.size(); ++i) {
            upper_bds.emplace_back(self().make_upper_bd());
        }
        upper_bd_t upper_bd_full = self().make_upper_bd();

        // routine to run on each partition
        auto run_partition = [&](size_t p_batch_size_) {

            // MUST reset the final upper bound object
            // and reset the range object.
            upper_bd_full.reset(max_lmda_size, p_batch_size_);
            curr_range.set_size(p_batch_size_);

            // run every thread
            auto lmda_subset = lmda_grid.head(max_lmda_size);
            auto thr_routine_wrapper = 
                [this](int thr_i,
                   size_t n_sim_per_thr,
                   size_t start_seed,
                   const rectangular_range_t& curr_range,
                   const auto& lmda_subset,
                   auto rng_gen_f,
                   upper_bd_t& upper_bd) 
                { 
                    thread_routine(
                            thr_i, n_sim_per_thr, start_seed, 
                            curr_range, lmda_subset, 
                            rng_gen_f, upper_bd);
                };

            for (size_t thr_i = 0; thr_i < pool.size()-1; ++thr_i) {
                pool[thr_i] = std::thread(
                        thr_routine_wrapper, 
                        thr_i, n_sim_per_thr, start_seed, 
                        std::cref(curr_range),
                        lmda_subset, rng_gen_f,
                        std::ref(upper_bds[thr_i]) );
            } 
            pool.back() = std::thread(
                        thr_routine_wrapper, 
                        pool.size()-1, n_sim_per_thr+n_sim_remain, start_seed, 
                        std::cref(curr_range), 
                        lmda_subset, rng_gen_f,
                        std::ref(upper_bds.back()) );

            // increment indexer while all threads are running 
            // pre-emptive update: optimization
            prev_range.set_idxer(curr_range.get_idxer());
            auto& curr_idxer = curr_range.get_idxer();
            for (size_t k = 0; k < p_batch_size_; ++k, ++curr_idxer);

            // wait until all threads finish
            for (auto& thr : pool) { thr.join(); }

            // pool upper bound estimates across all threads
            for (size_t i = 0; i < upper_bds.size(); ++i) {
                upper_bd_full.pool(upper_bds[i]);
            }

            process_upper_bd_f(
                    upper_bd_full, prev_range, p_endpt,
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

    template <class PVecType
            , class PEndptType
            , class RNGGenFType
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
            lmda_grid, rng_gen_f, start_seed, p_batch_size, pb, do_progress_bar, n_thr,
            process_upper_bd);
    }

    template <class PVecType
            , class PEndptType
            , class LmdaGridType
            , class RNGGenFType
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
            size_t start_seed=time(0),
            size_t p_batch_size=std::numeric_limits<size_t>::infinity(),
            PBType&& pb = PBType(std::cout),
            bool do_progress_bar = true,
            unsigned int n_thr=std::thread::hardware_concurrency()
            )
    {
        auto process_upper_bd = [](auto& upper_bd_full,
                                   const auto& p_range,
                                   const auto& p_endpt,
                                   auto alpha,
                                   auto thr_delta,
                                   auto grid_radius,
                                   auto& max_lmda_size) {
            // create the final upper bound
            upper_bd_full.create(
                    p_range, p_endpt,
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
                lmda_grid, rng_gen_f, start_seed, p_batch_size, pb, do_progress_bar, n_thr,
                process_upper_bd);
    }
};

} // namespace kevlar
