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
template <class ValueType, class UIntType, class GridRangeType>
struct ExpControlkTreatment;

namespace internal {

template <class T>
struct traits;

template <class ValueType, class UIntType, class GridRangeType>
struct traits<ExpControlkTreatment<ValueType, UIntType, GridRangeType>> {
    using value_t = ValueType;
    using uint_t = UIntType;
    using grid_range_t = GridRangeType;
    using state_t =
        typename ExpControlkTreatment<value_t, uint_t, grid_range_t>::StateType;
};

}  // namespace internal

/* Specialization declaration: rectangular grid */
template <class ValueType, class UIntType, class GridRangeType>
struct ExpControlkTreatment : ControlkTreatmentBase,
                              ModelBase<ValueType, UIntType, GridRangeType> {
   private:
    using static_interface_t = ControlkTreatmentBase;

   public:
    using value_t = ValueType;
    using uint_t = UIntType;
    using grid_range_t = GridRangeType;
    using base_t = static_interface_t;
    using model_base_t = ModelBase<ValueType, UIntType, GridRangeType>;

    struct StateType : ModelStateBase<value_t, uint_t, grid_range_t> {
       private:
        using outer_t = ExpControlkTreatment;
        const outer_t& outer_;

       public:
        using model_state_base_t =
            ModelStateBase<value_t, uint_t, grid_range_t>;

        StateType(const outer_t& outer)
            : model_state_base_t(outer),
              outer_(outer),
              exp_dist_(1.0),
              exp_(outer.n_samples(), outer.n_arms()),
              logrank_cum_sum_(2 * outer.n_samples() + 1),
              v_cum_sum_(2 * outer.n_samples() + 1) {}

        template <class GenType>
        void gen_rng(GenType&& gen) {
            exp_ = exp_.NullaryExpr(outer_.n_samples(), outer_.n_arms(),
                                    [&](auto, auto) { return exp_dist_(gen); });
        }

        /*
         * Overrides the necessary RNG generator.
         * TODO: generalize mt19937 somehow.
         */
        void gen_rng(std::mt19937& gen) override {
            gen_rng<std::mt19937&>(gen);
        }

        void gen_suff_stat() override {
            suff_stat_ = exp_.colwise().sum();
            sort_cols(exp_);
        }

        void rej_len(Eigen::Ref<colvec_type<uint_t>> rej_len) override {
            value_t hzrd_rate_prev = 1.0;   // hazard rate used previously
            bool do_logrank_update = true;  // true iff exp_ changed
            const auto& gr_view = outer_.grid_range();

            size_t pos = 0;
            for (size_t i = 0; i < gr_view.n_gridpts(); ++i) {
                if (gr_view.is_regular(i)) {
                    rej_len[pos] = (likely(gr_view.check_null(pos, 0)))
                                       ? rej_len_internal(i, hzrd_rate_prev,
                                                          do_logrank_update)
                                       : 0;
                    ++pos;
                    continue;
                }

                bool internal_called = false;
                size_t rej = 0;
                for (size_t t = 0; t < gr_view.n_tiles(i); ++t, ++pos) {
                    bool is_null = gr_view.check_null(pos, 0);
                    if (!internal_called && is_null) {
                        rej = rej_len_internal(i, hzrd_rate_prev,
                                               do_logrank_update);
                        internal_called = true;
                    }
                    rej_len[pos] = is_null ? rej : 0;
                }
            }
        }

        value_t grad(uint_t gridpt, uint_t arm) override {
            auto hazard_rate = outer_.hzrd_rate(gridpt);
            auto mean = (arm == 1) ? 1. / hazard_rate : 1.;
            auto lambda_control = outer_.lmda_control(gridpt);
            return (suff_stat_(arm) - outer_.n_samples() * mean) /
                   lambda_control;
        }

       private:
        KEVLAR_STRONG_INLINE
        size_t rej_len_internal(size_t i, value_t& hzrd_rate_prev,
                                bool& do_logrank_update) {
            auto hzrd_rate_curr = outer_.hzrd_rate(i);
            auto exp_control = exp_.col(0);    // assumed to be sorted
            auto exp_treatment = exp_.col(1);  // assumed to be sorted

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
                int cr_idx = 0, tr_idx = 0,
                    cs_idx = 0;  // control, treatment, and cum_sum index

                while (cr_idx < exp_control.size() &&
                       tr_idx < exp_treatment.size()) {
                    bool failed_in_treatment =
                        (exp_treatment[tr_idx] < exp_control[cr_idx]);
                    tr_idx += failed_in_treatment;
                    cr_idx += (1 - failed_in_treatment);
                    O_1j = failed_in_treatment;

                    auto N = N_j.sum();
                    auto E_1j = N_j[1] / N;
                    logrank_cum_sum_[cs_idx + 1] =
                        logrank_cum_sum_[cs_idx] + (O_1j - E_1j);
                    v_cum_sum_[cs_idx + 1] =
                        v_cum_sum_[cs_idx] + E_1j * (1 - E_1j);

                    --N_j[failed_in_treatment];
                    O_1j = 0.0;
                    ++cs_idx;
                }

                size_t tot = logrank_cum_sum_.size();
                logrank_cum_sum_.tail(tot - cs_idx).array() =
                    logrank_cum_sum_[cs_idx];
                v_cum_sum_.tail(tot - cs_idx).array() = v_cum_sum_[cs_idx];
            }

            // compute the log-rank statistic given the treatment lambda value.

            auto lambda_control = outer_.lmda_control(i);
            auto censor_dilated_curr = outer_.censor_time_ * lambda_control;
            auto it_c = std::lower_bound(
                exp_control.data(), exp_control.data() + exp_control.size(),
                censor_dilated_curr);
            auto it_t =
                std::lower_bound(exp_treatment.data(),
                                 exp_treatment.data() + exp_treatment.size(),
                                 censor_dilated_curr);
            // Y_1 Y_2 ...
            // T C T (censor) T T T C
            // idx = (2-1) + (3-1) = 3;
            size_t idx = std::distance(exp_control.data(), it_c) +
                         std::distance(exp_treatment.data(), it_t) - 2;
            auto z = (v_cum_sum_[idx] <= 0.0)
                         ? std::copysign(1., logrank_cum_sum_[idx]) *
                               std::numeric_limits<value_t>::infinity()
                         : logrank_cum_sum_[idx] / std::sqrt(v_cum_sum_[idx]);

            auto it = std::find_if(
                outer_.thresholds_.data(),
                outer_.thresholds_.data() + outer_.thresholds_.size(),
                [&](auto t) { return z > t; });
            return outer_.n_models() -
                   std::distance(outer_.thresholds_.data(), it);
        }

        std::exponential_distribution<value_t> exp_dist_;

        mat_type<value_t>
            exp_;  // exp_(i,j) =
                   //      Exp(1) draw for patient i in group j=0 (and sorted)
                   //      Exp(hzrd_rate) draw for patient i in group j=1 (and
                   //      sorted)
                   // We do not divide by lambda_control
                   // because log-rank only depends on the hazard rate.

        Eigen::Matrix<value_t, 1, 2>
            suff_stat_;  // sufficient statistic for each arm
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
        size_t n_samples, value_t censor_time,
        const Eigen::Ref<const colvec_type<value_t>>& thresholds)
        : base_t(2, 0, n_samples), censor_time_(censor_time) {
        set_thresholds(thresholds);
    }

    /*
     * Sets the grid range and caches any results
     * to speed-up the simulations.
     *
     * @param   grid_range      range of grid points.
     *                          0th dim = log(lambda_control)
     *                          1st dim = hazard rate (log(lambda_treatment /
     * lambda_control))
     *
     */
    void set_grid_range(const grid_range_t& grid_range) {
        model_base_t::set_grid_range(grid_range);

        n_gridpts_ = grid_range.n_gridpts();

        buff_.resize(n_arms() * n_gridpts_ * 2);

        auto pars = params();
        pars.array() = grid_range.thetas().array().exp();

        auto pars_lower = params_lower();
        pars_lower.array() =
            (grid_range.thetas() - grid_range.radii()).array().exp();
    }

    /*
     * Create a state object associated with the current model instance.
     */
    std::unique_ptr<typename state_t::model_state_base_t> make_state()
        const override {
        return std::make_unique<state_t>(*this);
    }

    uint_t n_models() const override { return thresholds_.size(); }

    value_t cov_quad(size_t j, const Eigen::Ref<const colvec_type<value_t>>& v)
        const override {
        auto hr = hzrd_rate(j);
        auto mean_1 = 1. / lmda_control(j);
        return n_samples() * mean_1 * mean_1 *
               (v[1] * v[1] + v[0] * v[0] / (hr * hr));
    }

    value_t max_cov_quad(
        size_t j,
        const Eigen::Ref<const colvec_type<value_t>>& v) const override {
        auto lmda_lower = lmda_control_lower(j);
        auto hr_lower = hzrd_rate_lower(j);
        auto mean_1 = 1. / lmda_lower;
        return n_samples() * mean_1 * mean_1 *
               (v[1] * v[1] + v[0] * v[0] / (hr_lower * hr_lower));
    }

    /*
     * Set the critical thresholds.
     */
    void set_thresholds(const Eigen::Ref<const colvec_type<value_t>>& thrs) {
        thresholds_ = thrs;
        std::sort(thresholds_.data(), thresholds_.data() + thresholds_.size(),
                  std::greater<value_t>());
    }

    /*
     * Sets the internal structure with the parameters.
     * Users should not interact with this method.
     * It is exposed purely for internal purposes (pickling).
     */
    void set_internal(uint_t n_gridpts, const colvec_type<value_t>& buff) {
        n_gridpts_ = n_gridpts;
        buff_ = buff;
    }

    /* Getter routines mainly for pickling */
    auto censor_time__() const { return censor_time_; }
    auto n_gridpts__() const { return n_gridpts_; }
    const auto& buff__() const { return buff_; }
    const auto& thresholds__() const { return thresholds_; }

   private:
    auto params_lower() {
        return Eigen::Map<mat_type<value_t>>(buff_.data(), n_arms(),
                                             n_gridpts_);
    }
    auto params_lower() const {
        return Eigen::Map<const mat_type<value_t>>(buff_.data(), n_arms(),
                                                   n_gridpts_);
    }
    auto params() {
        return Eigen::Map<mat_type<value_t>>(
            buff_.data() + n_gridpts_ * n_arms(), n_arms(), n_gridpts_);
    }
    auto params() const {
        return Eigen::Map<const mat_type<value_t>>(
            buff_.data() + n_gridpts_ * n_arms(), n_arms(), n_gridpts_);
    }
    auto hzrd_rate(size_t j) const { return params()(1, j); }
    auto hzrd_rate_lower(size_t j) const { return params_lower()(1, j); }
    auto lmda_control(size_t j) const { return params()(0, j); }
    auto lmda_control_lower(size_t j) const { return params_lower()(0, j); }

    const value_t censor_time_;
    colvec_type<value_t> thresholds_;
    uint_t n_gridpts_ = 0;
    colvec_type<value_t>
        buff_;  // buff_(0,j,0) = lower lambda of control at jth gridpoint.
                // buff_(1,j,0) = lower hazard rate at jth gridpoint.
                // buff_(0,j,1) = lambda of control at jth gridpoint.
                // buff_(1,j,1) = hazard rate at jth gridpoint.
};

}  // namespace kevlar
