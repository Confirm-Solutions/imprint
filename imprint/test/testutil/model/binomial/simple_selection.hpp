#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <imprint_bits/model/base.hpp>
#include <imprint_bits/util/algorithm.hpp>
#include <imprint_bits/util/d_ary_int.hpp>
#include <imprint_bits/util/math.hpp>
#include <imprint_bits/util/types.hpp>
#include <limits>
#include <random>
#include <testutil/model/base.hpp>

namespace imprint {
namespace model {
namespace binomial {
namespace legacy {

/*
 * Legacy BCKT class for testing purposes.
 */
struct BinomialControlkTreatment : ControlkTreatmentBase {
   private:
    using base_t = ControlkTreatmentBase;
    using static_interface_t = base_t;

   public:
    struct StateType {
       private:
        using outer_t = BinomialControlkTreatment;
        const outer_t& outer_;

       public:
        StateType(const outer_t& outer) : outer_(outer), unif_dist_(0., 1.) {}

        /*
         * Generate RNG for the simulation.
         * Generate U(0,1) for each patient in each arm.
         */
        template <class GenType>
        void gen_rng(GenType&& gen) {
            static_interface_t::uniform(outer_.n_samples(), outer_.n_arms(),
                                        gen, unif_dist_, unif_);
        }

        /*
         * Generates sufficient statistic for each arm under all possible grid
         * points.
         */
        void gen_suff_stat() {
            size_t k = outer_.n_arms() - 1;
            size_t n = outer_.n_samples();
            size_t ph2_size = outer_.ph2_size_;
            size_t ph3_size = n - ph2_size;
            size_t d = outer_.prob_.size();

            // grab the block of uniforms associated with Phase II/III for
            // treatments.
            auto ph2_unif = unif_.block(0, 1, ph2_size, k);
            auto ph3_unif = unif_.block(ph2_size, 1, ph3_size, k);

            // grab control uniforms
            auto control_unif = unif_.col(0);

            // sort each column of each block.
            sort_cols(ph2_unif);
            sort_cols(ph3_unif);
            sort_cols(control_unif);

            suff_stat_.resize(d, k + 1);
            Eigen::Map<Eigen::VectorXi> control_counts(suff_stat_.data(), d);
            Eigen::Map<Eigen::MatrixXi> ph3_counts(suff_stat_.col(1).data(), d,
                                                   k);
            ph2_counts_.resize(d, k);

            // output cumulative count of uniforms < outer_.prob_[k] into counts
            // object.
            accum_count(ph2_unif, outer_.prob_, ph2_counts_);
            accum_count(ph3_unif, outer_.prob_, ph3_counts);
            accum_count(control_unif, outer_.prob_, control_counts);

            suff_stat_.block(0, 1, d, k) += ph2_counts_;
        }

        /*
         * @param   mean_idxer      indexer of 1-d grid to get current grid
         * point (usually dAryInt).
         */
        template <class MeanIdxerType>
        auto test_stat(const MeanIdxerType& mean_idxer) const {
            auto& idx = mean_idxer();

            // Phase II
            int a_star =
                -1;  // selected arm with highest Phase II response count.
            int max_count = -1;  // maximum Phase II response count.
            for (int j = 1; j < idx.size(); ++j) {
                int prev_count = max_count;
                max_count = std::max(prev_count, ph2_counts_(idx[j], j - 1));
                a_star = (max_count != prev_count) ? j : a_star;
            }

            // Phase III

            // Only want false-rejection for Type-I.
            // Since the test is one-sided (upper), set to -inf if selected arm
            // is not in null.
            bool is_selected_arm_in_null =
                outer_.hypos_[a_star - 1](mean_idxer);
            if (!is_selected_arm_in_null)
                return -std::numeric_limits<double>::infinity();

            // pairwise z-test
            auto n = outer_.n_samples();
            auto p_star =
                static_cast<double>(suff_stat_(idx[a_star], a_star)) / n;
            auto p_0 = static_cast<double>(suff_stat_(idx[0], 0)) / n;
            auto z = (p_star - p_0);
            auto var = (p_star * (1. - p_star) + p_0 * (1. - p_0));
            z = (var <= 0) ? std::copysign(1.0, z) *
                                 std::numeric_limits<double>::infinity()
                           : z / std::sqrt(var / n);

            return z;
        }

        /*
         * Computes the gradient of the log-likelihood ratio:
         *      T - \nabla_\eta A(\eta)
         * where T is the sufficient statistic (vector), A is the log-partition
         * function, and \eta is the natural parameter.
         *
         * @param   arm             arm index.
         * @param   mean_idxer      indexer of 1-d grid to get current grid
         * point (usually dAryInt).
         */
        template <class MeanIdxerType>
        auto grad_lr(size_t arm, const MeanIdxerType& mean_idxer) const {
            auto& bits = mean_idxer();
            return suff_stat_(bits[arm], arm) -
                   outer_.n_samples() * outer_.prob_[bits[arm]];
        }

       private:
        std::uniform_real_distribution<double> unif_dist_;
        Eigen::MatrixXd unif_;        // uniform rng
        Eigen::MatrixXi suff_stat_;   // sufficient statistic table for each
                                      // prob_ value and arm
        Eigen::MatrixXi ph2_counts_;  // sufficient statistic table only looking
                                      // at phase 2 and treatment arms
    };

    using state_t = StateType;

    // @param   n_arms      number of arms.
    // @param   ph2_size    phase II size.
    // @param   n_samples   number of patients in each arm.
    // @param   prob        vector of (center) probability param to binomial.
    // MUST be sorted ascending.
    // @param   prob_endpt  each column is lower and upper of the grid centered
    // at prob.
    // @param   hypos       hypos[i](p) returns true if and only if
    //                      ith arm at prob value p is considered "in the null
    //                      space".
    template <class ProbType, class ProbEndptType>
    BinomialControlkTreatment(
        size_t n_arms, size_t ph2_size, size_t n_samples, const ProbType& prob,
        const ProbEndptType& prob_endpt,
        const std::vector<std::function<bool(const dAryInt&)> >& hypos)
        : base_t(n_arms, ph2_size, n_samples),
          prob_(prob.data(), prob.size()),
          prob_endpt_(prob_endpt.data(), prob_endpt.rows(), prob_endpt.cols()),
          hypos_(hypos) {}

    auto n_means() const { return prob_.size(); }

    constexpr auto n_total_params() const {
        return ipow(prob_.size(), n_arms());
    }

    /*
     * Computes the trace of the covariance matrix.
     * TODO: For now, this is all what upper-bound object requires, but may need
     * generalizing.
     *
     * @param   mean_idxer      indexer of 1-d grid to get current grid point
     * (usually dAryInt).
     */
    template <class MeanIdxerType>
    auto tr_cov(const MeanIdxerType& mean_idxer) const {
        const auto& bits = mean_idxer();
        const auto& p = prob_;
        double var = 0;
        std::for_each(bits.data(), bits.data() + bits.size(),
                      [&](auto k) { var += p[k] * (1. - p[k]); });
        return var * n_samples();
    }

    /*
     * Computes the trace of the supremum (in the grid) of covariance matrix.
     * TODO: For now, this is all what upper-bound object requires, but may need
     * generalizing.
     *
     * @param   mean_idxer      indexer of 1-d grid to get current grid point
     * (usually dAryInt).
     */
    template <class MeanIdxerType>
    auto tr_max_cov(const MeanIdxerType& mean_idxer) const {
        double hess_bd = 0;
        const auto& bits = mean_idxer();
        std::for_each(bits.data(), bits.data() + bits.size(), [&](auto k) {
            auto col_k = prob_endpt_.col(k);
            if (col_k[0] <= 0.5 && 0.5 <= col_k[1]) {
                hess_bd += 0.25;
            } else {
                auto lower = col_k[0] - 0.5;  // shift away center
                auto upper = col_k[1] - 0.5;  // shift away center
                // max of p(1-p) occurs for whichever p is closest to 0.5.
                bool max_at_upper = (std::abs(upper) < std::abs(lower));
                auto max_endpt = col_k[max_at_upper];
                hess_bd += max_endpt * (1. - max_endpt);
            }
        });
        return hess_bd * n_samples();
    }

   private:
    Eigen::Map<const Eigen::VectorXd>
        prob_;  // sorted (ascending) probability values
    Eigen::Map<const Eigen::MatrixXd>
        prob_endpt_;  // each column is endpt (in p-space) of the grid
                      // centered at the corresponding value in prob_
    const std::vector<std::function<bool(const dAryInt&)> >&
        hypos_;  // list of null-hypothesis checker
};

}  // namespace legacy
}  // namespace binomial
}  // namespace model
}  // namespace imprint
