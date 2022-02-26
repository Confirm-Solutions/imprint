#pragma once
#include <kevlar_bits/util/types.hpp>
#include <kevlar_bits/util/algorithm.hpp>
#include <kevlar_bits/util/math.hpp>
#include <kevlar_bits/util/macros.hpp>
#include <kevlar_bits/model/base.hpp>
#include <Eigen/Core>
#include <limits>
#include <algorithm>
#include <random>

namespace kevlar {

/* Forward declaration */
template <class ValueType, class UIntType>
struct ExpControlkTreatment;

namespace internal {

template <class T>
struct traits;

template <class ValueType, class UIntType>
struct traits<ExpControlkTreatment<ValueType, UIntType> >
{
    using value_t = ValueType;
    using uint_t = UIntType;
    using state_t = typename ExpControlkTreatment<value_t, uint_t>::StateType;
};

} // namespace internal

/* Specialization declaration: rectangular grid */
template <class ValueType, class UIntType>
struct ExpControlkTreatment
    : ControlkTreatmentBase
    , ModelBase<ValueType>
{
private:
    using base_t = ControlkTreatmentBase;
    using static_interface_t = base_t;

public:
    using value_t = ValueType;
    using uint_t = UIntType;

    struct StateType : ModelStateBase<value_t, uint_t>
    {
    private:
        using outer_t = ExpControlkTreatment;
        const outer_t& outer_;

    public:
        StateType(const outer_t& outer)
            : outer_(outer)
            , exp_dist_(1.0)
            , exp_(outer.n_samples(), outer.n_arms())
            , logrank_cum_sum_(2*outer.n_samples()+1)
            , v_cum_sum_(2*outer.n_samples()+1)
        {}

        template <class GenType>
        void gen_rng(GenType&& gen) { 
            exp_ = exp_.NullaryExpr(outer_.n_samples(), outer_.n_arms(),
                    [&](auto, auto) { return exp_dist_(gen); });
        }

        void gen_suff_stat() {
            suff_stat_ = exp_.colwise().sum();
            sort_cols(exp_);
        }

        void get_rej_len(Eigen::Ref<colvec_type<uint_t> > rej_len) override  
        {
            bool do_logrank_update = true;     // true iff exp_ changed
            value_t hzrd_rate_prev = 1.0;       // hazard rate used previously

            for (size_t i = 0; i < outer_.n_gridpts(); ++i) {

                // if current gridpoint is not in null hypothesis
                if (unlikely(!outer_.null_hypo_[i])) {
                    rej_len[i] = 0;
                    continue;
                }

                auto hzrd_rate_curr = outer_.hzrd_rate(i);
                auto exp_control = exp_.col(0);     // assumed to be sorted
                auto exp_treatment = exp_.col(1);   // assumed to be sorted

                // Since log-rank test only depends on hazard-rate,
                // we can reuse the same pre-computed quantities for all lambdas.
                // We only update internal quantities if we see a new hazard rate.
                // Performance is best if the gridpoints are grouped by
                // the same hazard rate so that the internals are not updated often.
                if (hzrd_rate_curr != hzrd_rate_prev) {
                    auto hzrd_rate_ratio = (hzrd_rate_prev / hzrd_rate_curr);

                    // compute treatment ~ Exp(hzrd_rate_curr)
                    exp_treatment *= hzrd_rate_ratio;
                    suff_stat_[1] *= hzrd_rate_ratio;

                    // if hzrd rate was different from previous run,
                    // save the current one as the new "previous"
                    hzrd_rate_prev = hzrd_rate_curr;

                    // since exp_ has been updated
                    do_logrank_update = true;
                }

                // compute log-rank information only if exp_ changed
                if (do_logrank_update) {
                    // mark as not needing update
                    do_logrank_update = false;

                    logrank_cum_sum_[0] = 0.0;
                    v_cum_sum_[0] = 0.0;

                    Eigen::Matrix<value_t, 2, 1> N_j;
                    value_t O_1j = 0.0;
                    N_j.array() = outer_.n_samples();
                    int cr_idx = 0, tr_idx = 0, i=0; // control, treatment, and cum_sum index

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

                // compute the log-rank statistic given the treatment lambda value.

                auto lambda_control = outer_.lmda_control(i);
                auto censor_dilated_curr = outer_.censor_time_ * lambda_control;
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
                        std::copysign(1., logrank_cum_sum_[idx]) * std::numeric_limits<value_t>::infinity() :
                        logrank_cum_sum_[idx] / std::sqrt(v_cum_sum_[idx]);

                auto it = std::find_if(
                        outer_.thresholds_.data(),
                        outer_.thresholds_.data()+outer_.thresholds_.size(),
                        [&](auto t) { return z > t; });
                rej_len[i] = outer_.n_models() - std::distance(outer_.thresholds_.data(), it);
            }
        }

        value_t get_grad(uint_t gridpt, uint_t arm) override
        {
            auto hazard_rate = outer_.hzrd_rate(gridpt);
            auto mean = (arm == 1) ? 1./hazard_rate : 1.;
            auto lambda_control = outer_.lmda_control(gridpt);
            return (suff_stat_(arm) - outer_.n_samples() * mean) / lambda_control;
        }

        constexpr auto n_models() const { return outer_.n_models(); }
        constexpr auto n_gridpts() const { return outer_.n_gridpts(); }
        constexpr auto n_params() const { return outer_.n_arms(); }

    private:
        std::exponential_distribution<value_t> exp_dist_; 
    
        mat_type<value_t> exp_;             // exp_(i,j) = 
                                            //      Exp(1) draw for patient i in group j=0 (and sorted)
                                            //      Exp(hzrd_rate) draw for patient i in group j=1 (and sorted)
                                            // We do not divide by lambda_control 
                                            // because log-rank only depends on the hazard rate.
                                            
        Eigen::Matrix<value_t,1,2> suff_stat_; // sufficient statistic for each arm
                                               // - sum of Exp(1) for group 0 (control)
                                               // - sum of Exp(hzrd_rate) for group 1 (treatment)
        colvec_type<value_t> logrank_cum_sum_;
        colvec_type<value_t> v_cum_sum_;
    };

    using state_t = StateType;

    // default constructor is the base constructor
    using base_t::base_t;

    // @param   n_samples       number of patients in each arm.
    // @param   censor_time     censor time.
    ExpControlkTreatment(
            size_t n_samples,
            value_t censor_time,
            const Eigen::Ref<const colvec_type<value_t> >& thresholds)
        : base_t(2, 0, n_samples)
        , censor_time_(censor_time)
    {
        set_thresholds(thresholds);
    }

    /*
     * Sets the grid range and caches any results
     * to speed-up the simulations.
     *
     * @param   grid_range      range of grid points.
     *                          0th dim = log(lambda_control)
     *                          1st dim = hazard rate (log(lambda_treatment / lambda_control))
     * @param   null_hypo       functor that checks if the ith arm is in 
     *                          its null-hypothesis given a gridpoint.
     *
     */
    template <class GridRangeType, class NullHypoType>
    void set_grid_range(
            const GridRangeType& grid_range,
            const NullHypoType& null_hypo) 
    {
        n_gridpts_ = grid_range.size();

        buff_.resize(n_arms() * n_gridpts_ * 2);

        auto params = get_params();
        params.array() = grid_range.get_thetas().array().exp();

        auto params_lower = get_params_lower();
        params_lower.array() = (grid_range.get_thetas() - grid_range.get_radii()).array().exp();

        null_hypo_.resize(n_gridpts_);
        for (size_t i = 0; i < n_gridpts_; ++i) {
            null_hypo_[i] = null_hypo(params.col(i)); 
        }
    }

    /*
     * Create a state object associated with the current model instance.
     */
    state_t make_state() const { return state_t(*this); }

    constexpr auto n_models() const { return thresholds_.size(); }
    constexpr auto n_gridpts() const { return n_gridpts_; }

    value_t cov_quad(size_t j, const Eigen::Ref<const colvec_type<value_t>>& v) const override
    {
        auto hr = hzrd_rate(j);
        auto mean_1 = 1./lmda_control(j);
        return n_samples() * mean_1 * mean_1 *
            (v[1]*v[1] + v[0]*v[0]/(hr * hr));
    }

    value_t max_cov_quad(size_t j, const Eigen::Ref<const colvec_type<value_t>>& v) const override
    {
        auto lmda_lower = lmda_control_lower(j);
        auto hr_lower = hzrd_rate_lower(j);
        auto mean_1 = 1./lmda_lower;
        return n_samples() * mean_1 * mean_1 *
            (v[1]*v[1] + v[0]*v[0]/(hr_lower*hr_lower));
    }

    /*
     * Set the critical thresholds.
     */
    void set_thresholds(const Eigen::Ref<const colvec_type<value_t>>& thrs) 
    {
        thresholds_ = thrs;
        std::sort(thresholds_.data(), 
                  thresholds_.data()+thresholds_.size(), 
                  std::greater<value_t>());
    }

    /*
     * Sets the internal structure with the parameters.
     * Users should not interact with this method.
     * It is exposed purely for internal purposes (pickling).
     */
    void set_internal(
            uint_t n_gridpts,
            const colvec_type<value_t>& buff,
            const std::vector<bool>& null_hypo)
    {
        n_gridpts_ = n_gridpts;
        buff_ = buff;
        null_hypo_ = null_hypo;
    }

    /* Getter routines mainly for pickling */
    auto get_censor_time() const { return censor_time_; }
    auto get_n_gridpts() const { return n_gridpts_; }
    const auto& get_buff() const { return buff_; }
    const auto& get_null_hypo() const { return null_hypo_; }
    const auto& get_thresholds() const { return thresholds_; }

private:
    auto get_params_lower() {
        return Eigen::Map<mat_type<value_t>>(
                buff_.data(),
                n_arms(), n_gridpts_);
    }
    auto get_params_lower() const {
        return Eigen::Map<const mat_type<value_t>>(
                buff_.data(),
                n_arms(), n_gridpts_);
    }
    auto get_params() {
        return Eigen::Map<mat_type<value_t>>(
                buff_.data() + n_gridpts_ * n_arms(),
                n_arms(), n_gridpts_);
    }
    auto get_params() const {
        return Eigen::Map<const mat_type<value_t>>(
                buff_.data() + n_gridpts_ * n_arms(),
                n_arms(), n_gridpts_);
    }
    auto hzrd_rate(size_t j) const { return get_params()(1,j); }
    auto hzrd_rate_lower(size_t j) const { return get_params_lower()(1,j); }
    auto lmda_control(size_t j) const { return get_params()(0,j); }
    auto lmda_control_lower(size_t j) const { return get_params_lower()(0,j); }

    const value_t censor_time_;
    colvec_type<value_t> thresholds_;
    uint_t n_gridpts_ = 0;
    colvec_type<value_t> buff_;     // buff_(0,j,0) = lower lambda of control at jth gridpoint.
                                    // buff_(1,j,0) = lower hazard rate at jth gridpoint.
                                    // buff_(0,j,1) = lambda of control at jth gridpoint.
                                    // buff_(1,j,1) = hazard rate at jth gridpoint.
    std::vector<bool> null_hypo_;   // null_hypo_(p) = true 
                                    // if p is in the null hypothesis.
};

} // namespace kevlar
