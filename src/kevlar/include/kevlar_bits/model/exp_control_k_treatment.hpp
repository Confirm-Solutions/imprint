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
            , exp_(outer.n_samples(), outer.n_arms())
            , logrank_cum_sum_(2*outer.n_samples()+1)
            , v_cum_sum_(2*outer.n_samples()+1)
        {}

        template <class GenType>
        void gen_rng(GenType&& gen) { 
            exp_ = exp_.NullaryExpr(outer_.n_samples(), outer_.n_arms(),
                    [&](auto, auto) { 
                        return std::exponential_distribution<double>(1.0)(gen);
                    });
            exp_.col(1) /= hzrd_rate_prev_;
            do_logrank_update_ = true;
        }

        void gen_suff_stat() {
            suff_stat_ = exp_.colwise().sum();
            sort_cols(exp_);
        }

        template <class IdxerType>
        auto test_stat(const IdxerType& idxer) {
            auto& bits = idxer(); 
            auto hzrd_rate_curr = outer_.hzrd_rate_[bits[0]];
            auto exp_control = exp_.col(0);     // assumed to be sorted
            auto exp_treatment = exp_.col(1);   // assumed to be sorted

            if (hzrd_rate_curr != hzrd_rate_prev_) {
                auto hzrd_rate_ratio = (hzrd_rate_prev_ / hzrd_rate_curr);

                // compute treatment ~ Exp(hzrd_rate_curr)
                exp_treatment *= hzrd_rate_ratio;
                suff_stat_[1] *= hzrd_rate_ratio;

                // if hzrd rate was different from previous run,
                // save the current one as the new "previous"
                hzrd_rate_prev_ = hzrd_rate_curr;

                // since exp_ has been updated
                do_logrank_update_ = true;
            }

            // compute log-rank information only if exp_ changed
            if (do_logrank_update_) {
                // mark as not needing update
                do_logrank_update_ = false;

                logrank_cum_sum_[0] = 0.0;
                v_cum_sum_[0] = 0.0;

                Eigen::Matrix<double, 2, 1> N_j;
                double O_1j = 0.0;
                N_j.array() = outer_.n_samples();
                size_t cr_idx = 0, tr_idx = 0, i=0; // control, treatment, and cum_sum index

                while (cr_idx < exp_control.size() && tr_idx < exp_treatment.size()) 
                {
                    bool failed_in_treatment = (exp_treatment[tr_idx] < exp_control[cr_idx]);
                    tr_idx += failed_in_treatment;
                    cr_idx += (1-failed_in_treatment);
                    O_1j = failed_in_treatment;

                    auto N = N_j.sum();
                    auto E_1j = N_j[1] / N;
                    logrank_cum_sum_[i+1] = logrank_cum_sum_[i] + (O_1j - E_1j);
                    v_cum_sum_[i+1] = v_cum_sum_[i] + E_1j * (1-E_1j);
                    
                    --N_j[failed_in_treatment];
                    O_1j = 0.0;
                    ++i;
                }

                size_t tot = logrank_cum_sum_.size();
                logrank_cum_sum_.tail(tot-i).array() = logrank_cum_sum_[i];
                v_cum_sum_.tail(tot-i).array() = v_cum_sum_[i];
            }

            auto censor_dilated_curr = outer_.censor_time_ * outer_.lmda_[bits[1]];
            auto it_c = std::lower_bound(exp_control.data(), 
                                         exp_control.data()+exp_control.size(), 
                                         censor_dilated_curr);
            auto it_t = std::lower_bound(exp_treatment.data(), 
                                         exp_treatment.data()+exp_treatment.size(), 
                                         censor_dilated_curr);
            // Y_1 Y_2 ...
            // T C T (censor) T T T C
            // idx = (2-1) + (3-1) = 3;
            size_t idx = std::distance(exp_control.data(), it_c) +
                         std::distance(exp_treatment.data(), it_t) - 2;
            auto z = (v_cum_sum_[idx] <= 0.0) ? 
                    std::copysign(1., logrank_cum_sum_[idx]) * std::numeric_limits<double>::infinity() :
                    logrank_cum_sum_[idx] / std::sqrt(v_cum_sum_[idx]);
            return z;
        }

        template <class IdxerType>
        auto grad_lr(size_t arm, const IdxerType& idxer) const {
            auto& bits = idxer();
            auto mean = (arm == 1) ? 1./hzrd_rate_prev_ : 1.;
            return (suff_stat_(arm) - outer_.n_samples() * mean) / outer_.lmda_[bits[1]];
        }

    private:
        bool do_logrank_update_ = true;     // true iff exp_ changed
        double hzrd_rate_prev_ = 1.0;       // hazard rate used previously
        Eigen::MatrixXd exp_;               // exp_(i,j) = 
                                            //      Exp(1) draw for patient i in group j=0 (and sorted)
                                            //      Exp(hzrd_rate_prev_) draw for patient i in group j=1 (and sorted)
        Eigen::Vector2d suff_stat_;         // sufficient statistic for each arm
                                            // - sum of Exp(1) for group 0 (control)
                                            // - sum of Exp(hzrd_rate_prev_) for group 1 (treatment)
        Eigen::VectorXd logrank_cum_sum_;
        Eigen::VectorXd v_cum_sum_;
    };

    using state_t = StateType;

    // @param   n_samples       number of patients in each arm
    // @param   lmda            grid of lmda (ascending) (must be evenly-spaced in natural param space)
    //                          Must not include 0.
    // @param   lmda_lower      lower boundary point of rectangle (in natural param space) centered at lmda 
    //                          Must not include 0.
    // @param   hzrd_rate       exp(offset) where offset is evenly-spaced (same radius as lmda in natural param).
    //                          Must not include 0.
    // @param   hzrd_rate_lower lower boundary point of rectangle (in offset space) centered at hzrd_rate.
    //                          Must not include 0.
    template <class LmdaType, class LmdaLowerType, class HzrdType, class HzrdLowerType>
    ExpControlkTreatment(
            size_t n_samples,
            double censor_time,
            const LmdaType& lmda,
            const LmdaLowerType& lmda_lower,
            const HzrdType& hzrd_rate,
            const HzrdLowerType& hzrd_rate_lower)
        : base_t(2, 0, n_samples)
        , censor_time_(censor_time)
        , lmda_(lmda.data(), lmda.size())
        , lmda_lower_(lmda_lower.data(), lmda_lower.rows(), lmda_lower.cols())
        , hzrd_rate_(hzrd_rate.data(), hzrd_rate.size())
        , hzrd_rate_lower_(hzrd_rate_lower.data(), hzrd_rate_lower.size())
    {}

    Eigen::Index n_means() const { return lmda_.size(); }

    constexpr auto n_total_params() const { 
        assert(lmda_.size() == hzrd_rate_.size());
        return lmda_.size() * hzrd_rate_.size();
    }

    template <class IdxerType>
    auto tr_cov(const IdxerType& idxer) const
    {
        auto& bits = idxer();
        auto hzrd_rate = hzrd_rate_[bits[0]];
        auto mean_1 = 1./lmda_[bits[1]];
        auto mean_0 = 1./hzrd_rate * mean_1;
        return n_samples() * (mean_1*mean_1 + mean_0*mean_0);
    }

    template <class IdxerType>
    auto tr_max_cov(const IdxerType& idxer) const
    {
        auto& bits = idxer();
        auto lmda_lower = lmda_lower_(bits[1]);
        auto hzrd_rate_lower = hzrd_rate_lower_(bits[0]);
        return n_samples() * (1./(lmda_lower*lmda_lower)) * (1.0 + 1./(hzrd_rate_lower*hzrd_rate_lower));
    }

private:
    double censor_time_;
    Eigen::Map<const Eigen::VectorXd> lmda_;        // sorted (ascending) lmda (center) values
    Eigen::Map<const Eigen::VectorXd> lmda_lower_;  // lower boundary point in lmda_ grid
    Eigen::Map<const Eigen::VectorXd> hzrd_rate_;   // sorted (ascending) hazard rate
    Eigen::Map<const Eigen::VectorXd> hzrd_rate_lower_; // lower boundary point in hzrd_rate_ grid
};

} // namespace kevlar
