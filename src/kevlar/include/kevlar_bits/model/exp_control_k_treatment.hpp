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
struct ExpControlkTreatment;

namespace internal {

template <class T>
struct traits;

template <class GridType>
struct traits<ExpControlkTreatment<GridType> >
{
    using state_t = typename ExpControlkTreatment<GridType>::StateType;
};

} // namespace internal

/* Specialization declaration: rectangular grid */
template <>
struct ExpControlkTreatment<grid::Rectangular>
    : ControlkTreatmentBase
    , ModelBase<ExpControlkTreatment<grid::Rectangular> >
{
private:
    using base_t = ControlkTreatmentBase;
    using static_interface_t = base_t;

public:
    struct StateType 
    {
    private:
        using outer_t = ExpControlkTreatment;
        const outer_t& outer_;

    public:
        StateType(const outer_t& outer)
            : outer_(outer)
        {}

        template <class GenType>
        void gen_rng(GenType&& gen) { 
            std::gamma_distribution<double> g(outer_.n_samples_, 1.);            
            std::gamma_distribution<double> g_ph2(outer_.ph2_size_, 1.);            
            std::gamma_distribution<double> g_ph3(outer_.n_samples_-outer_.ph2_size_, 1.);            

            gamma_control_ = g(gen);
            gamma_.resize(outer_.n_arms_-1, 2);
            gamma_.col(0) = Eigen::VectorXd::NullaryExpr(gamma_.rows(), 
                    [&](auto) { return g_ph2(gen); });
            gamma_.col(1) = Eigen::VectorXd::NullaryExpr(gamma_.rows(), 
                    [&](auto) { return g_ph3(gen); });
        }

        void gen_suff_stat() {
            size_t k = outer_.n_arms_-1;
            size_t d = outer_.n_means();

            suff_stat_.resize(d, k+1);
            Eigen::Map<Eigen::VectorXd> control_suff(suff_stat_.data(), d);
            Eigen::Map<Eigen::MatrixXd> ph3_suff(suff_stat_.col(1).data(), d, k);
            ph2_suff_stat_.resize(d, k);

            control_suff.array() = gamma_control_ / outer_.lmda_.array();
            ph3_suff = (1./outer_.lmda_.array()).matrix() * gamma_.col(1).transpose();
            ph2_suff_stat_ = (1./outer_.lmda_.array()).matrix() * gamma_.col(0).transpose();
            suff_stat_.block(0, 1, d, k) += ph2_suff_stat_;
        }

        template <class MeanIdxerType>
        auto test_stat(const MeanIdxerType& mean_idxer) const {
            auto& idx = mean_idxer();

            // Phase II
            int a_star = -1;
            int max_count = -1;
            for (int j = 1; j < idx.size(); ++j) {
                int prev_count = max_count;
                max_count = std::max(prev_count, ph2_suff_stat_(idx[j], j-1));
                a_star = (max_count != prev_count) ? j : a_star;
            }

            // Phase III
            auto n = outer_.n_samples_;
            auto p_star = static_cast<double>(suff_stat_(idx[a_star], a_star)) / n;
            auto p_0 = static_cast<double>(suff_stat_(idx[0], 0)) / n;
            auto z = (p_star - p_0);
            auto var = (p_star * (1.-p_star) + p_0 * (1.-p_0));
            z = (var <= 0) ? 
                std::copysign(1.0, z) * std::numeric_limits<double>::infinity() : 
                z / std::sqrt(var / n); 
            return z;
        }

        auto grad_lr(size_t arm, size_t mean_idx) const {
            return suff_stat_(mean_idx, arm) - outer_.n_samples_ * outer_.prob_[mean_idx];
        }

    private:
        double gamma_control_ = 0;          // gamma rng for arm 0
        Eigen::MatrixXd gamma_;             // gamma_(i,j) = gamma rng for phase (j+2) and arm (i+1)
        Eigen::MatrixXd suff_stat_;         // sufficient statistic table for each prob_ value and arm
        Eigen::MatrixXd ph2_suff_stat_;     // sufficient statistic table only looking at phase 2 and treatment arms
    };

    using state_t = StateType;

    // @param   prob        MUST be sorted.
    ExpControlkTreatment(
            size_t n_arms,
            size_t ph2_size,
            size_t n_samples,
            const Eigen::VectorXd& lmda,
            const Eigen::MatrixXd& lmda_endpt)
        : base_t(n_arms, ph2_size, n_samples)
        , lmda_(lmda)
        , lmda_endpt_(lmda_endpt)
    {}

    Eigen::Index n_means() const { return lmda_.size(); }

    constexpr auto n_total_params() const { 
        return ipow(lmda_.size(), n_arms_);
    }

    template <class MeanIdxerType>
    auto tr_cov(const MeanIdxerType& mean_idxer) const
    {
        const auto& bits = mean_idxer();
        const auto& l = lmda_;
        double var = 0;
        std::for_each(bits.data(), bits.data() + bits.size(),
            [&](auto k) { var += l[k] * (1.-p[k]); });
        return var * n_samples_;
    }

    template <class MeanIdxerType>
    auto tr_max_cov(const MeanIdxerType& mean_idxer) const
    {
        double hess_bd = 0;
        const auto& bits = mean_idxer();
        std::for_each(bits.data(), bits.data() + bits.size(),
            [&](auto k) {
                auto col_k = prob_endpt_.col(k);
                auto lower = col_k[0] - 0.5; // shift away center
                auto upper = col_k[1] - 0.5; // shift away center
                // max of p(1-p) occurs for whichever p is closest to 0.5.
                bool max_at_upper = (std::abs(upper) < std::abs(lower));
                auto max_endpt = col_k[max_at_upper]; 
                hess_bd += max_endpt * (1. - max_endpt);
            });
        return hess_bd * n_samples_;
    }

private:
    const Eigen::VectorXd& lmda_;       // sorted (ascending) probability values
    const Eigen::MatrixXd& lmda_endpt_; // each column is endpt (in p-space) of the grid centered at the corresponding value in prob_
};

} // namespace kevlar
