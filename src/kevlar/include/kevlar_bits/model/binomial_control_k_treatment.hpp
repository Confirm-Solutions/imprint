#pragma once
#include <kevlar_bits/util/types.hpp>
#include <kevlar_bits/util/algorithm.hpp>
#include <kevlar_bits/model/base.hpp>
#include <kevlar_bits/model/control_k_treatment_base.hpp>
#include <Eigen/Core>
#include <limits>
#include <algorithm>

namespace kevlar {

/* Forward declaration */
template <class GridType = grid::Arbitrary>
struct BinomialControlkTreatment;

namespace internal {

template <class T>
struct traits;

template <class GridType>
struct traits<BinomialControlkTreatment<GridType> >
{
    using state_t = typename BinomialControlkTreatment<GridType>::StateType;
};

} // namespace internal

/*
 * Binomial control + k Treatment model.
 * For a given point null p = (p_0,..., p_{k}),
 * n responses Y_{ij} for each arm j=0,...,k where Y_{ij} ~ Bern(p_j) iid,
 * Phase II size of ph2_size, it does the following procedure:
 *
 *  - select the treatment arm j* with most responses based on the first ph2_size samples
 *  - construct the paired z-test between p_{j*} and p_0 testing for the null that p_{j*} <= p_0.
 */

/* Specialization declaration: arbitrary grid */
template <>
struct BinomialControlkTreatment<grid::Arbitrary>
    : ControlkTreatmentBase
{
private:
    using base_t = ControlkTreatmentBase;

public:
    using base_t::base_t;

    /*
     * Runs the Binomial 3-arm Phase II/III trial simulation.
     * 
     * @param   unif            matrix with n_arms columns where column i is the uniform draws of arm i.
     * @param   p_range         current range of the full p-grid.
     * @param   thr_grid        Grid of threshold values for tuning. See requirements for UpperBound.
     * @param   upper_bd        Upper-bound object to update.
     */
    template <class UnifType
            , class PRangeType
            , class ThrVecType
            , class UpperBoundType>
    inline void run(
            const UnifType& unif,
            const PRangeType& p_range,
            const ThrVecType& thr_grid,
            UpperBoundType& upper_bd
            ) const
    {
        assert(static_cast<size_t>(unif.rows()) == n_samples());
        assert(static_cast<size_t>(unif.cols()) == n_arms());

        size_t k = unif.cols()-1;
        size_t n = unif.rows();
        size_t ph2_size = ph2_size;
        size_t ph3_size = n - ph2_size;

        // resize cache
        Eigen::VectorXd a_sum(k);

        auto z_stat = [&](const auto& p) {
            // phase II
            for (size_t i = 0; i < a_sum.size(); ++i) {
                a_sum[i] = (unif.col(i+1).head(ph2_size).array() < p(i+1)).count();
            }

            // compare and choose arm with more successes
            Eigen::Index a_star;
            a_sum.maxCoeff(&a_star);

            // phase III
            size_t a_star_rest_sum = (unif.col(a_star+1).tail(ph3_size).array() < p(a_star+1)).count();
            auto p_star = static_cast<double>(a_sum[a_star] + a_star_rest_sum) / n;
            auto p_0 = (unif.col(0).array() < p(0)).template cast<double>().mean();
            auto z = (p_star - p_0);
            auto var = (p_star * (1.-p_star) + p_0 * (1.-p_0));
            z = (var == 0) ? std::numeric_limits<double>::infinity() : z / sqrt(var / n); 
            return z;
        };
    }
};

/* Specialization declaration: rectangular grid */
template <>
struct BinomialControlkTreatment<grid::Rectangular>
    : ControlkTreatmentBase
    , ModelBase<BinomialControlkTreatment<grid::Rectangular> >
{
private:
    using base_t = ControlkTreatmentBase;
    using static_interface_t = base_t;

public:
    struct StateType 
    {
    private:
        using outer_t = BinomialControlkTreatment;
        const outer_t& outer_;

    public:
        StateType(const outer_t& outer)
            : outer_(outer)
        {}

        template <class GenType>
        void gen_rng(GenType&& gen) { 
            static_interface_t::uniform(0., 1., gen, unif_, outer_.n_samples(), outer_.n_arms()); 
        }

        void gen_suff_stat() {
            size_t k = outer_.n_arms()-1;
            size_t n = outer_.n_samples();
            size_t ph2_size = outer_.ph2_size_;
            size_t ph3_size = n - ph2_size;
            size_t d = outer_.prob_.size();

            auto ph2_unif = unif_.block(0, 1, ph2_size, k);
            auto ph3_unif = unif_.block(ph2_size, 1, ph3_size, k);
            auto control_unif = unif_.col(0);
            sort_cols(ph2_unif);
            sort_cols(ph3_unif);
            sort_cols(control_unif);

            suff_stat_.resize(d, k+1);
            Eigen::Map<Eigen::VectorXi> control_counts(suff_stat_.data(), d);
            Eigen::Map<Eigen::MatrixXi> ph3_counts(suff_stat_.col(1).data(), d, k);
            ph2_counts_.resize(d, k);
            cum_count(ph2_unif, outer_.prob_, ph2_counts_);
            cum_count(ph3_unif, outer_.prob_, ph3_counts);
            cum_count(control_unif, outer_.prob_, control_counts);
            suff_stat_.block(0, 1, d, k) += ph2_counts_;
        }

        template <class MeanIdxerType>
        auto test_stat(const MeanIdxerType& mean_idxer) const {
            auto& idx = mean_idxer();

            // Phase II
            int a_star = -1;
            int max_count = -1;
            for (int j = 1; j < idx.size(); ++j) {
                int prev_count = max_count;
                max_count = std::max(prev_count, ph2_counts_(idx[j], j-1));
                a_star = (max_count != prev_count) ? j : a_star;
            }

            // Phase III

            // Only want false-rejection for Type-I
            // Since the test is one-sided (upper), set to -inf if selected arm is not in null.
            bool is_selected_arm_in_null = outer_.hypos_[a_star-1](mean_idxer);
            if (!is_selected_arm_in_null) return -std::numeric_limits<double>::infinity();

            auto n = outer_.n_samples();
            auto p_star = static_cast<double>(suff_stat_(idx[a_star], a_star)) / n;
            auto p_0 = static_cast<double>(suff_stat_(idx[0], 0)) / n;
            auto z = (p_star - p_0);
            auto var = (p_star * (1.-p_star) + p_0 * (1.-p_0));
            z = (var <= 0) ? 
                std::copysign(1.0, z) * std::numeric_limits<double>::infinity() : 
                z / std::sqrt(var / n); 

            return z;
        }

        template <class MeanIdxerType>
        auto grad_lr(size_t arm, const MeanIdxerType& mean_idxer) const {
            auto& bits = mean_idxer();
            return suff_stat_(bits[arm], arm) - outer_.n_samples() * outer_.prob_[bits[arm]];
        }

    private:
        Eigen::MatrixXd unif_;              // uniform rng
        Eigen::MatrixXi suff_stat_;         // sufficient statistic table for each prob_ value and arm
        Eigen::MatrixXi ph2_counts_;        // sufficient statistic table only looking at phase 2 and treatment arms
    };

    using state_t = StateType;

    // @param   n_arms      number of arms.
    // @param   ph2_size    phase II size.
    // @param   n_samples   number of patients in each arm.
    // @param   prob        vector of (center) probability param to binomial. MUST be sorted ascending.
    // @param   prob_endpt  each column is lower and upper of the grid centered at prob.
    // @param   hypos       hypos[i](p) returns true if and only if 
    //                      ith arm at prob value p is considered "in the null space".
    template <class ProbType, class ProbEndptType> 
    BinomialControlkTreatment(
            size_t n_arms,
            size_t ph2_size,
            size_t n_samples,
            const ProbType& prob,
            const ProbEndptType& prob_endpt,
            const std::vector<std::function<bool(const dAryInt&)> >& hypos)
        : base_t(n_arms, ph2_size, n_samples)
        , prob_(prob.data(), prob.size())
        , prob_endpt_(prob_endpt.data(), prob_endpt.rows(), prob_endpt.cols())
        , hypos_(hypos)
    {}

    auto n_means() const { return prob_.size(); }

    constexpr auto n_total_params() const { 
        return ipow(prob_.size(), n_arms());
    }

    template <class MeanIdxerType>
    auto tr_cov(const MeanIdxerType& mean_idxer) const
    {
        const auto& bits = mean_idxer();
        const auto& p = prob_;
        double var = 0;
        std::for_each(bits.data(), bits.data() + bits.size(),
            [&](auto k) { var += p[k] * (1.-p[k]); });
        return var * n_samples();
    }

    template <class MeanIdxerType>
    auto tr_max_cov(const MeanIdxerType& mean_idxer) const
    {
        double hess_bd = 0;
        const auto& bits = mean_idxer();
        std::for_each(bits.data(), bits.data() + bits.size(),
            [&](auto k) {
                auto col_k = prob_endpt_.col(k);
                if (col_k[0] <= 0.5 && 0.5 <= col_k[1]) {
                    hess_bd += 0.25;
                } else {
                    auto lower = col_k[0] - 0.5; // shift away center
                    auto upper = col_k[1] - 0.5; // shift away center
                    // max of p(1-p) occurs for whichever p is closest to 0.5.
                    bool max_at_upper = (std::abs(upper) < std::abs(lower));
                    auto max_endpt = col_k[max_at_upper]; 
                    hess_bd += max_endpt * (1. - max_endpt);
                }
            });
        return hess_bd * n_samples();
    }

private:
    Eigen::Map<const Eigen::VectorXd> prob_;        // sorted (ascending) probability values
    Eigen::Map<const Eigen::MatrixXd> prob_endpt_;  // each column is endpt (in p-space) of the grid 
                                                    // centered at the corresponding value in prob_
    const std::vector<std::function<bool(const dAryInt&)> >& hypos_;    // list of null-hypothesis checker
};

} // namespace kevlar
