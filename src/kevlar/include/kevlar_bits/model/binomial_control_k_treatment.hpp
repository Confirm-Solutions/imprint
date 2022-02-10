#pragma once
#include <kevlar_bits/util/types.hpp>
#include <kevlar_bits/util/algorithm.hpp>
#include <kevlar_bits/util/d_ary_int.hpp>
#include <kevlar_bits/util/math.hpp>
#include <kevlar_bits/model/base.hpp>
#include <limits>
#include <algorithm>
#include <set>
#include <random>

namespace kevlar {

/* Forward declaration */
template <class ValueType, class IntType>
struct BinomialControlkTreatment;

namespace internal {

template <class T>
struct traits;

template <class ValueType, class IntType>
struct traits<BinomialControlkTreatment<ValueType, IntType> >
{
    using value_t = ValueType;
    using int_t = IntType;
    using state_t = typename BinomialControlkTreatment<ValueType, IntType>::StateType;
};

} // namespace internal

template <class ValueType, class IntType>
struct BinomialControlkTreatment
    : ControlkTreatmentBase
{
private:
    using base_t = ControlkTreatmentBase;
    using static_interface_t = base_t;

public:
    using value_t = ValueType;
    using int_t = IntType;

    struct StateType : ModelStateBase<value_t, int_t>
    {
    private:
        using outer_t = BinomialControlkTreatment;
        const outer_t& outer_;

    public:
        StateType(const outer_t& outer)
            : outer_(outer)
            , unif_dist_(0., 1.)
        {
            for (auto& pu : outer_.probs_unique_) {
                n_total_uniques_ += pu.size();
            }
        }

        /*
         * Generate RNG for the simulation.
         * Generate U(0,1) for each patient in each arm.
         * TODO: what's the interface?
         */
        template <class GenType>
        void gen_rng(GenType&& gen) { 
            static_interface_t::uniform(
                    outer_.n_samples(), outer_.n_arms(),  
                    gen, unif_dist_, unif_);
        }

        /*
         * Generates sufficient statistic for each arm under all possible grid points.
         */
        void gen_suff_stat() 
        {
            size_t k = outer_.n_arms()-1;
            size_t n = outer_.n_samples();
            size_t ph2_size = outer_.ph2_size_;
            size_t ph3_size = n - ph2_size;

            // grab the block of uniforms associated with Phase II/III for treatments.
            auto ph2_unif = unif_.block(0, 1, ph2_size, k);
            auto ph3_unif = unif_.block(ph2_size, 1, ph3_size, k);

            // grab control uniforms
            auto control_unif = unif_.col(0);

            // sort each column of each block.
            sort_cols(ph2_unif);
            sort_cols(ph3_unif);
            sort_cols(control_unif);

            suff_stat_.resize(n_total_uniques_);
            Eigen::Map<colvec_type<int_t> > control_counts(
                    suff_stat_.data(), outer_.probs_unique_[0].size());
            Eigen::Map<colvec_type<int_t> > ph3_counts(
                    suff_stat_.data() + control_counts.size(), suff_stat_.size() - control_counts.size());
            ph2_counts_.resize(ph3_counts.size());

            // output cumulative count of uniforms < outer_.prob_[k] into counts object.
            cum_count(control_unif, outer_.probs_unique_[0], control_counts);
            {
                size_t n_skip = 0;
                for (int i = 0; i < k; ++i) {
                    Eigen::Map<colvec_type<int_t> > ph2_counts_i(
                            ph2_counts_.data() + n_skip, outer_.probs_unique_[i+1].size());
                    Eigen::Map<colvec_type<int_t> > ph3_counts_i(
                            ph3_counts.data() + n_skip, outer_.probs_unique_[i+1].size());
                    cum_count(ph2_unif.col(i), outer_.probs_unique_[i+1], ph2_counts_i);
                    cum_count(ph3_unif.col(i), outer_.probs_unique_[i+1], ph3_counts_i);

                    n_skip += ph2_counts_i.size();
                }
            }

            // final update on suff_stat
            suff_stat_.tail(suff_stat_.size()-control_counts.size()) += ph2_counts_;
        }

        /*
         * @param   rej_len      vector of integers to store number of models that reject.
         */ 
        void get_rej_len(Eigen::Ref<colvec_type<int_t> > rej_len) const override 
        {
            auto& bits = outer_.gbits_;

            for (int i = 0; i < outer_.n_gridpts(); ++i) 
            {
                auto bits_i = bits.col(i);

                // Phase II
                int a_star = -1;    // selected arm with highest Phase II response count.
                int max_count = -1; // maximum Phase II response count.
                for (int j = 1; j < bits_i.size(); ++j) {
                    int prev_count = max_count;
                    Eigen::Map<const colvec_type<int_t> > ph2_counts_v(
                            ph2_counts_.data() + outer_.strides_[j] - outer_.strides_[1],
                            outer_.strides_[j+1] - outer_.strides_[j]);
                    max_count = std::max(
                            static_cast<int64_t>(prev_count), 
                            static_cast<int64_t>(ph2_counts_v(bits_i[j])) );
                    a_star = (max_count != prev_count) ? j : a_star;
                }

                // Phase III
                
                // pairwise z-test
                auto n = outer_.n_samples();
                Eigen::Map<const colvec_type<int_t> > ss_astar(
                        suff_stat_.data() + outer_.strides_[a_star],
                        outer_.strides_[a_star+1] - outer_.strides_[a_star]);
                Eigen::Map<const colvec_type<int_t> > ss_0(
                        suff_stat_.data(),
                        outer_.strides_[1]);
                auto p_star = static_cast<double>(ss_astar(bits_i[a_star])) / n;
                auto p_0 = static_cast<double>(ss_0(bits_i[0])) / n;
                auto z = (p_star - p_0);
                auto var = (p_star * (1.-p_star) + p_0 * (1.-p_0));
                z = (var <= 0) ? 
                    std::copysign(1.0, z) * std::numeric_limits<double>::infinity() : 
                    z / std::sqrt(var / n); 

                // save rejection for the current model
                rej_len[i] = (z > outer_.threshold_);
            }
        }

        /*
         * Computes the gradient of the log-likelihood ratio:
         *      T - \nabla_\eta A(\eta)
         * where T is the sufficient statistic (vector), A is the log-partition function, and \eta is the natural parameter.
         *
         * @param   grad             flattened 2-d array where grad(i,j) = at gridpt i, jth element of (T-\nabla A).
         */
        void get_grad(Eigen::Ref<colvec_type<value_t> > grad) const override
        {
            Eigen::Map<mat_type<value_t> > grad_m(grad.data(), outer_.n_gridpts(), outer_.n_arms());
            auto& bits = outer_.gbits_;
            for (int j = 0; j < grad_m.cols(); ++j) {
                Eigen::Map<const colvec_type<int_t> > ss_a(
                        suff_stat_.data() + outer_.strides_(j),
                        outer_.strides_(j+1) - outer_.strides_(j));
                for (int i = 0; i < grad_m.rows(); ++i) {
                    grad_m(i,j) = ss_a(bits(j,i)) - outer_.n_samples() * outer_.p_(j,i);
                }
            } 
        }

    private:
        std::uniform_real_distribution<value_t> unif_dist_;
        mat_type<value_t> unif_;          // uniform rng
        colvec_type<int_t> suff_stat_;    // sufficient statistic table for each arm and prob value
                                          // suff_stat_(i,j) = suff stat at unique prob i at arm j.
        colvec_type<int_t> ph2_counts_;   // sufficient statistic table only looking at phase 2 and treatment arms
                                          // ph2_counts_(i,j) = ph2 suff stat at unique prob i at arm j.
        size_t n_total_uniques_ = 0;
    };

    using state_t = StateType;

    // @param   n_arms      number of arms.
    // @param   ph2_size    phase II size.
    // @param   n_samples   number of patients in each arm.
    // @param   grid_range  Range of gridpts in natural parameter space.
    template <class GridRangeType> 
    BinomialControlkTreatment(
            size_t n_arms,
            size_t ph2_size,
            size_t n_samples,
            const GridRangeType& grid_range,
            value_t threshold)
        : base_t(n_arms, ph2_size, n_samples)
        , probs_unique_(n_arms)
        , strides_(n_arms+1)
        , probs_(n_arms * grid_range.size() * 3)
        , p_(probs_.data() + n_arms * grid_range.size(), n_arms, grid_range.size())
        , gbits_(n_arms, grid_range.size())
        , threshold_(threshold)
    {
        strides_[0] = 0;

        auto& thetas = grid_range.get_thetas();

        // populate prob matrix
        auto& radii = grid_range.get_radii();
        Eigen::Map<mat_type<value_t> > pm(probs_.data(), n_arms, grid_range.size());
        pm = thetas - radii;
        new (&pm) Eigen::Map<mat_type<value_t> >(pm.data() + pm.size(), pm.rows(), pm.cols());
        pm = thetas;
        new (&pm) Eigen::Map<mat_type<value_t> >(pm.data() + pm.size(), pm.rows(), pm.cols());
        pm = thetas + radii;
        probs_.array() = sigmoid(probs_.array());

        // populate set of unique theta values for each arm
        std::unordered_map<value_t, int_t> pu_to_idx;
        std::set<value_t> prob_set;
        auto& bits = gbits_;

        for (size_t i = 0; i < n_arms; ++i) {
            pu_to_idx.clear();
            prob_set.clear();

            // insert all prob values in arm i into the set
            auto prob_row = p_.row(i);
            for (int j = 0; j < prob_row.size(); ++j) {
                prob_set.insert(prob_row[j]);
            }

            // create a mapping from unique prob values to order idx
            {
                int j = 0;
                for (auto p : prob_set) {
                    pu_to_idx[p] = j++;
                }
            }

            // copy number of uniques
            strides_[i+1] = strides_[i] + prob_set.size();

            // copy unique prob values into vector
            probs_unique_[i].resize(prob_set.size());
            std::copy(prob_set.begin(),
                      prob_set.end(),
                      probs_unique_[i].data());

            // populate bits for current arm
            auto bits_i = bits.row(i);
            for (int j = 0; j < bits_i.size(); ++j) {
                bits_i(j) = pu_to_idx[p_(i,j)];
            }
        }
    }

    constexpr auto n_gridpts() const { return p_.cols(); }

    /*
     * Computes the trace of the covariance matrix at ith gridpoint.
     *
     * @param   i      ith gridpoint.
     */
    value_t tr_cov(size_t i) const
    {
        auto pi = p_.col(i).array();
        return (pi * (1.0 - pi)).sum() * n_samples();
    }

    /*
     * Computes the trace of the supremum (in the grid) of covariance matrix at ith gridpoint.
     *
     * @param   i      ith gridpoint.
     */
    value_t tr_max_cov(size_t i) const
    {
        Eigen::Map<const mat_type<value_t> > p_lower(
                probs_.data(), p_.rows(), p_.cols());
        Eigen::Map<const mat_type<value_t> > p_upper(
                p_.data() + p_.size(), p_.rows(), p_.cols());
        auto pli = p_lower.col(i);
        auto pi = p_.col(i);
        auto pui = p_upper.col(i);

        value_t hess_bd = 0;
        for (int j = 0; j < pi.size(); ++j) {
            if (pli[j] <= 0.5 && 0.5 <= pui[j]) {
                hess_bd += 0.25;
            } else {
                auto lower = pli[j] - 0.5; // shift away center
                auto upper = pui[j] - 0.5; // shift away center
                // max of p(1-p) occurs for whichever p is closest to 0.5.
                bool max_at_upper = (std::abs(upper) < std::abs(lower));
                auto max_endpt = max_at_upper ? pui[j] : pli[j]; 
                hess_bd += max_endpt * (1. - max_endpt);
            }
        }
        return hess_bd * n_samples();
    }

private:
    colvec_type<colvec_type<value_t> > probs_unique_; // probs_unique_[i] = unique prob vector sorted (ascending) for arm i.
    colvec_type<int_t> strides_;    // strides_[i] = number of unique probs for arm i-1 with 0 for arm -1.
    colvec_type<value_t> probs_;    // probs_(.,j,k) = jth prob vector with k = 0,1,2 corresp to left/curr/right gridpt.
    Eigen::Map<mat_type<value_t> > p_;   // viewer of 2nd slice in probs_ (current gridpts)
    mat_type<int_t> gbits_; // range of gbits
    value_t threshold_; // critical threshold
};

} // namespace kevlar
