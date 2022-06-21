#pragma once
#include <imprint_bits/util/macros.hpp>
#include <imprint_bits/util/types.hpp>

namespace imprint {
namespace stat {

template <class ValueType, class UIntType>
struct LogRankTest {
    using value_t = ValueType;
    using uint_t = UIntType;

   private:
    Eigen::Map<const colvec_type<value_t>> control_;    // sorted control
    Eigen::Map<const colvec_type<value_t>> treatment_;  // sorted treatment
    colvec_type<value_t> logrank_accum_;
    // logrank_accum_[i] =
    // log-rank test statistic
    // considering only the first i events
    // (either in control or treatment).

   public:
    /*
     * Constructs the object by storing references
     * to the control and treatment vector outcomes.
     * We assume control and treatment are sorted in ascending order.
     */
    template <class ControlType, class TreatmentType>
    LogRankTest(const ControlType& control, const TreatmentType& treatment)
        : control_(control.data(), control.size()),
          treatment_(treatment.data(), treatment.size()),
          logrank_accum_(control.size() + treatment.size() + 1) {}

    LogRankTest(const LogRankTest& other)
        : control_(other.control_.data(), other.control_.size()),
          treatment_(other.treatment_.data(), other.treatment_.size()),
          logrank_accum_(other.logrank_accum_) {}

    LogRankTest(LogRankTest&& other)
        : control_(other.control_.data(), other.control_.size()),
          treatment_(other.treatment_.data(), other.treatment_.size()),
          logrank_accum_(std::move(other.logrank_accum_)) {}

    LogRankTest& operator=(const LogRankTest& other) {
        new (&control_) Eigen::Map<const colvec_type<value_t>>(
            other.control_.data(), other.control_.size());
        new (&treatment_) Eigen::Map<const colvec_type<value_t>>(
            other.treatment_.data(), other.treatment_.size());
        logrank_accum_ = other.logrank_accum_;
    }

    LogRankTest& operator=(LogRankTest&& other) {
        new (&control_) Eigen::Map<const colvec_type<value_t>>(
            other.control_.data(), other.control_.size());
        new (&treatment_) Eigen::Map<const colvec_type<value_t>>(
            other.treatment_.data(), other.treatment_.size());
        logrank_accum_ = std::move(other.logrank_accum_);
    }

    /*
     * Runs the log-rank test and stores the cumulative log-rank test
     * statistics.
     */
    void run() {
        value_t logrank_cum_sum = 0.0;
        value_t v_cum_sum = 0.0;
        logrank_accum_[0] = 0.0;

        mat_type<uint_t, 2, 1> N_j;
        N_j[0] = control_.size();
        N_j[1] = treatment_.size();

        mat_type<uint_t, 2, 1> O_j;

        int cr_idx = 0, tr_idx = 0,
            cs_idx = 0;  // control, treatment, and cum_sum index

        auto count_outcomes = [](const auto& v, auto& idx, auto& counter) {
            auto idx_old = idx;
            auto v_old = v[idx_old];
            for (++idx; (idx < v.size()) && (v[idx] == v_old); ++idx)
                ;
            counter += idx - idx_old;
        };

        while (cr_idx < control_.size() && tr_idx < treatment_.size()) {
            // Reset current number of outcomes.
            O_j.array() = 0;

            // save these values since the next if-blocks
            // may advance the indices.
            auto c_val = control_[cr_idx];
            auto t_val = treatment_[tr_idx];

            // Computes the number of outcomes for treatment arm
            // and moves forward the indexer
            // if an outcome came first or at the same time as control arm.
            if (t_val <= c_val) {
                count_outcomes(treatment_, tr_idx, O_j[1]);
            }

            // Computes the number of outcomes for control arm
            // and moves forward the indexer
            // if an outcome came first or at the same time as treatment arm.
            if (t_val >= c_val) {
                count_outcomes(control_, cr_idx, O_j[0]);
            }

            // Compute accumulations
            // Note that this logic only well-defined if N > 1.
            // We do not have to check for if N > 1 since if
            //  * N == 0: no one "at risk"
            //  * N == 1: one "at risk" so one arm has no one "at risk"
            // In both cases, we would already be outside the loop from previous
            // iteration.
            uint_t N = N_j.sum();
            uint_t O = O_j.sum();
            value_t O_div_N = O / static_cast<value_t>(N);
            value_t E_0j = N_j[0] * O_div_N;
            logrank_cum_sum += (O_j[0] - E_0j);
            v_cum_sum +=
                E_0j * (1 - O_div_N) * (static_cast<value_t>(N_j[1]) / (N - 1));

            // Note that this may leave some values of logrank_accum_
            // uninitialized if there are repeat outcomes at a distinct time. We
            // just need to be careful when get the stat given a censor time.
            logrank_accum_[cs_idx + O] =
                (v_cum_sum <= 0.0)
                    ? std::copysign(1., logrank_cum_sum) *
                          std::numeric_limits<value_t>::infinity()
                    : logrank_cum_sum / std::sqrt(v_cum_sum);

            // Update number of subjects "at risk"
            N_j.array() -= O_j.array();

            // Increment accumulation indexer
            cs_idx += O;
        }

        // Optimization: the rest of the log-rank stats
        // are the same as the previous value.
        // This is because O_{ij} - E_{ij} = 0
        // for the distinct times j starting now,
        // so nothing accumulates anymore.
        size_t tot = logrank_accum_.size();
        logrank_accum_.tail(tot - cs_idx).array() = logrank_accum_[cs_idx];
    }

    /*
     * Returns the log-rank statistic given censor time as censor_time.
     * If an observation is exactly at censor_time,
     * it is counted towards the log-rank statistic.
     * See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC403858/,
     * which uses this convention.
     */
    IMPRINT_STRONG_INLINE
    value_t stat(value_t censor_time, bool control_based) const {
        // computes the number of observations in v <= censor_time
        auto n_observed = [&](const auto& v) {
            // find first time v outcome > censor_time
            auto it =
                std::upper_bound(v.data(), v.data() + v.size(), censor_time);
            return (it - v.data());
        };

        auto n_c_observed = n_observed(control_);
        auto n_t_observed = n_observed(treatment_);

        size_t idx = n_c_observed + n_t_observed;
        auto lr_stat = logrank_accum_[idx];
        return control_based ? lr_stat : -lr_stat;
    }
};

}  // namespace stat
}  // namespace imprint
