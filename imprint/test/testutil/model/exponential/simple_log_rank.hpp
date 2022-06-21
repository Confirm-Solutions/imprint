#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <imprint_bits/model/base.hpp>
#include <imprint_bits/util/algorithm.hpp>
#include <imprint_bits/util/macros.hpp>
#include <imprint_bits/util/math.hpp>
#include <imprint_bits/util/types.hpp>
#include <limits>
#include <random>
#include <testutil/model/base.hpp>

namespace imprint {
namespace model {
namespace exponential {
namespace legacy {

template <class ValueType, class UIntType, class GridRangeType>
struct ExpControlkTreatment : ControlkTreatmentBase,
                              ModelBase<ValueType>,
                              SimGlobalStateBase<ValueType, UIntType> {
   private:
    using static_interface_t = ControlkTreatmentBase;

   public:
    using value_t = ValueType;
    using uint_t = UIntType;
    using grid_range_t = GridRangeType;
    using base_t = static_interface_t;
    using model_base_t = ModelBase<ValueType>;
    using gen_t = std::mt19937;
    using sgs_t = SimGlobalStateBase<value_t, uint_t>;

    struct StateType : sgs_t::sim_state_t {
       private:
        using outer_t = ExpControlkTreatment;
        const outer_t& outer_;

       public:
        StateType(const outer_t& outer, size_t seed)
            : outer_(outer),
              exp_dist_(1.0),
              exp_(outer.n_samples(), outer.n_arms()),
              logrank_cum_sum_(2 * outer.n_samples() + 1),
              v_cum_sum_(2 * outer.n_samples() + 1),
              gen_(seed) {}

        void simulate(Eigen::Ref<colvec_type<uint_t>> rej_len) override {
            // generate data
            exp_ =
                exp_.NullaryExpr(outer_.n_samples(), outer_.n_arms(),
                                 [&](auto, auto) { return exp_dist_(gen_); });

            // generate suff stat
            suff_stat_ = exp_.colwise().sum();

            // sort for log-rank stuff
            sort_cols(exp_);

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

        void score(size_t gridpt,
                   Eigen::Ref<colvec_type<value_t>> out) const override {
            auto hazard_rate = outer_.hzrd_rate(gridpt);
            for (size_t arm = 0; arm < 2; ++arm) {
                auto mean = (arm == 1) ? 1. / hazard_rate : 1.;
                auto lambda_control = outer_.lmda_control(gridpt);
                out[arm] = (suff_stat_(arm) - outer_.n_samples() * mean) /
                           lambda_control;
            }
        }

       private:
        IMPRINT_STRONG_INLINE
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
            auto it_c = std::upper_bound(
                exp_control.data(), exp_control.data() + exp_control.size(),
                censor_dilated_curr);
            auto it_t =
                std::upper_bound(exp_treatment.data(),
                                 exp_treatment.data() + exp_treatment.size(),
                                 censor_dilated_curr);
            // Y_1 Y_2 ...
            // T C T (censor) T T T C
            // idx = (2-1) + (3-1) = 3;
            size_t idx = std::distance(exp_control.data(), it_c) +
                         std::distance(exp_treatment.data(), it_t);
            auto z = (v_cum_sum_[idx] <= 0.0)
                         ? std::copysign(1., logrank_cum_sum_[idx]) *
                               std::numeric_limits<value_t>::infinity()
                         : logrank_cum_sum_[idx] / std::sqrt(v_cum_sum_[idx]);

            auto it = std::find_if(outer_.critical_values().begin(),
                                   outer_.critical_values().end(),
                                   [&](auto t) { return z > t; });
            return std::distance(it, outer_.critical_values().end());
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
        gen_t gen_;
    };

    using state_t = StateType;

    // default constructor is the base constructor
    using base_t::base_t;

    // @param   n_samples       number of patients in each arm.
    // @param   censor_time     censor time.
    ExpControlkTreatment(
        size_t n_samples, value_t censor_time,
        const Eigen::Ref<const colvec_type<value_t>>& thresholds)
        : base_t(2, 0, n_samples),
          model_base_t(thresholds),
          max_eta_hess_cov_(3 * std::sqrt(n_samples)),
          censor_time_(censor_time) {
        // temporarily const-cast just to initialize the values
        auto& max_cov_nc_ = const_cast<mat_type<value_t, 2, 2>&>(max_cov_);
        max_cov_nc_.setOnes();
        max_cov_nc_(0, 0) = 2;
        max_cov_nc_ *= n_samples;
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
        grid_range_ = &grid_range;

        n_gridpts_ = grid_range.n_gridpts();

        buff_.resize(n_arms(), n_gridpts_);

        buff_.array() = grid_range.thetas().array().exp();
    }

    /*
     * Create a state object associated with the current model instance.
     */
    std::unique_ptr<typename sgs_t::sim_state_t> make_sim_state(
        size_t seed) const override {
        return std::make_unique<state_t>(*this, seed);
    }

    value_t cov_quad(size_t j,
                     const Eigen::Ref<const colvec_type<value_t>>& v) const {
        auto hr = hzrd_rate(j);
        auto mean_1 = 1. / lmda_control(j);
        return n_samples() * mean_1 * mean_1 *
               (v[1] * v[1] + v[0] * v[0] / (hr * hr));
    }

    value_t max_cov_quad(
        size_t, const Eigen::Ref<const colvec_type<value_t>>& v) const {
        return v.dot(max_cov_ * v);
    }

    /*
     * Deta = [
     *  [e^{\theta_1} 0]
     *  [e^{\theta_1 + \theta_2} e^{\theta_1 + \theta_2}]
     * ]
     * \theta_1 = \log(\lambda_c)
     * \theta_2 = \log(\lambda_t / \lambda_c)
     */
    void eta_transform(size_t j,
                       const Eigen::Ref<const colvec_type<value_t>>& v,
                       colvec_type<value_t>& out) const {
        value_t lmda_c = lmda_control(j);
        value_t lmda_t = lmda_c * hzrd_rate(j);

        mat_type<value_t, 2, 2> deta;
        deta(0, 0) = lmda_c;
        deta(0, 1) = 0;
        deta.row(1).array() = lmda_t;

        out = deta * v;
    }

    value_t max_eta_hess_cov(size_t) const { return max_eta_hess_cov_; }

    /*
     * Sets the internal structure with the parameters.
     * Users should not interact with this method.
     * It is exposed purely for internal purposes (pickling).
     */
    void set_internal(uint_t n_gridpts, const mat_type<value_t>& buff) {
        n_gridpts_ = n_gridpts;
        buff_ = buff;
    }

    /* Getter routines mainly for pickling */
    auto censor_time__() const { return censor_time_; }
    auto n_gridpts__() const { return n_gridpts_; }
    const auto& buff__() const { return buff_; }

   private:
    auto lmda_control(size_t j) const { return buff_(0, j); }
    auto hzrd_rate(size_t j) const { return buff_(1, j); }
    const auto& grid_range() const { return *grid_range_; }

    const grid_range_t* grid_range_;
    const value_t max_eta_hess_cov_;  // caches max_eta_hess_cov() result
    const value_t censor_time_;
    uint_t n_gridpts_ = 0;
    mat_type<value_t>
        buff_;  // buff_(0,j) = lambda of control at jth gridpoint.
                // buff_(1,j) = hazard rate at jth gridpoint.
    const mat_type<value_t, 2, 2> max_cov_;
};

}  // namespace legacy
}  // namespace exponential
}  // namespace model
}  // namespace imprint
