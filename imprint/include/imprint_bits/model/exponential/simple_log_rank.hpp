#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <imprint_bits/model/base.hpp>
#include <imprint_bits/model/exponential/common/fixed_n_log_hazard_rate.hpp>
#include <imprint_bits/model/fixed_single_arm_size.hpp>
#include <imprint_bits/stat/log_rank_test.hpp>
#include <imprint_bits/util/algorithm.hpp>
#include <imprint_bits/util/macros.hpp>
#include <imprint_bits/util/math.hpp>
#include <imprint_bits/util/types.hpp>
#include <random>

namespace imprint {
namespace model {
namespace exponential {

template <class ValueType>
struct SimpleLogRank : FixedSingleArmSize, ModelBase<ValueType> {
    using arm_base_t = FixedSingleArmSize;
    using base_t = ModelBase<ValueType>;
    using typename base_t::value_t;

   private:
    const value_t censor_time_;

    /*
     * Returns total number of parameters.
     * Simply an alias for n_arms() since there is 1 parameter per arm.
     */
    IMPRINT_STRONG_INLINE auto n_params() const { return n_arms(); }

   public:
    template <class _GenType, class _ValueType, class _UIntType,
              class _GridRangeType>
    struct SimGlobalState;

    template <class _GenType, class _ValueType, class _UIntType,
              class _GridRangeType>
    using sim_global_state_t =
        SimGlobalState<_GenType, _ValueType, _UIntType, _GridRangeType>;

    template <class _GridRangeType>
    using imprint_bound_state_t =
        ImprintBoundStateFixedNLogHazardRate<_GridRangeType>;

    SimpleLogRank(size_t n_arm_samples, value_t censor_time,
                  const Eigen::Ref<const colvec_type<value_t>>& cv)
        : arm_base_t(2, n_arm_samples), base_t(), censor_time_(censor_time) {
        critical_values(cv);
    }

    using arm_base_t::n_arm_samples;
    using arm_base_t::n_arms;

    using base_t::critical_values;
    void critical_values(const Eigen::Ref<const colvec_type<value_t>>& cv) {
        auto& cv_ = base_t::critical_values();
        cv_ = cv;
        std::sort(cv_.begin(), cv_.end(), std::greater<value_t>());
    }

    template <class _GenType, class _ValueType, class _UIntType,
              class _GridRangeType>
    auto make_sim_global_state(const _GridRangeType& grid_range) const {
        return sim_global_state_t<_GenType, _ValueType, _UIntType,
                                  _GridRangeType>(*this, grid_range);
    }

    template <class _GridRangeType>
    auto make_imprint_bound_state(const _GridRangeType& gr) const {
        return imprint_bound_state_t<_GridRangeType>(n_arm_samples(), gr);
    }

    // Extra model-specific functions
    auto censor_time() const { return censor_time_; }
};

template <class ValueType>
template <class _GenType, class _ValueType, class _UIntType,
          class _GridRangeType>
struct SimpleLogRank<ValueType>::SimGlobalState
    : SimGlobalStateFixedNLogHazardRate<_GenType, _ValueType, _UIntType,
                                        _GridRangeType> {
    struct SimState;

    using base_t = SimGlobalStateFixedNLogHazardRate<_GenType, _ValueType,
                                                     _UIntType, _GridRangeType>;
    using typename base_t::gen_t;
    using typename base_t::grid_range_t;
    using typename base_t::interface_t;
    using typename base_t::uint_t;
    using typename base_t::value_t;

    using sim_state_t = SimState;

   private:
    using model_t = SimpleLogRank;
    const model_t& model_;
    const grid_range_t& grid_range_;

   public:
    SimGlobalState(const model_t& model, const grid_range_t& grid_range)
        : base_t(model.n_arm_samples(), grid_range),
          model_(model),
          grid_range_(grid_range) {}

    std::unique_ptr<typename interface_t::sim_state_t> make_sim_state(
        size_t seed) const override {
        return std::make_unique<sim_state_t>(*this, seed);
    }
};

template <class ValueType>
template <class _GenType, class _ValueType, class _UIntType,
          class _GridRangeType>
struct SimpleLogRank<ValueType>::SimGlobalState<_GenType, _ValueType, _UIntType,
                                                _GridRangeType>::SimState
    : SimGlobalState::base_t::sim_state_t {
   private:
    using outer_t = SimGlobalState;

   public:
    using base_t = typename outer_t::base_t::sim_state_t;
    using typename base_t::interface_t;

   private:
    const outer_t& outer_;
    stat::LogRankTest<value_t, uint_t>
        lrt_;  // log-rank test fitter.
               // It is initialized with control ~ Exp(1),
               // treatment ~ Exp(hzrd_rate) since the test
               // only depends on hzrd_rate.
               // It views control, treatment vectors in base class.

    IMPRINT_STRONG_INLINE
    size_t rej_len_internal(size_t i) {
        // If it's the first grid-point, do logrank update.
        bool do_logrank_update = (i == 0);

        auto hzrd_rate_prev =
            base_t::hzrd_rate();  // previously saved hazard rate
        auto hzrd_rate_curr = outer_.hzrd_rate(i);  // current hazard rate

        // Since log-rank test only depends on hazard-rate,
        // we can reuse the same pre-computed quantities for all control
        // lambdas. We only update internal quantities if we see a new hazard
        // rate. Performance is best if the gridpoints are grouped by the same
        // hazard rate so that the internals are not updated often. Note that
        // lrt_ still internally points to the same control and treatment
        // storages. This is deliberate - we want lrt_ to reference these
        // updated storages.
        if (hzrd_rate_curr != hzrd_rate_prev) {
            base_t::update_hzrd_rate(hzrd_rate_curr);
            do_logrank_update = true;
        }

        // compute log-rank information only if needed
        if (do_logrank_update) {
            lrt_.run();
        }

        // Compute the log-rank statistic given the treatment lambda value.
        // Since lrt_ references Exp(1), Exp(hzrd_rate) vectors,
        // the censor time must be dilated to be on the same scale.
        auto lambda_control = outer_.lmda_control(i);
        auto censor_dilated_curr = outer_.model_.censor_time_ * lambda_control;
        auto z = lrt_.stat(censor_dilated_curr, false);

        const auto& cvs = outer_.model_.critical_values();
        int cv_i = 0;
        for (; cv_i < cvs.size(); ++cv_i) {
            if (z > cvs[cv_i]) {
                break;
            }
        }
        return cvs.size() - cv_i;
    }

   public:
    SimState(const outer_t& outer, size_t seed)
        : base_t(outer, seed),
          outer_(outer),
          lrt_(base_t::control(), base_t::treatment()) {}

    void simulate(Eigen::Ref<colvec_type<uint_t>> rej_len) override {
        base_t::generate_data();
        base_t::generate_sufficient_stats();

        // sort the columns to optimize log-rank procedure.
        sort_cols(base_t::control());
        sort_cols(base_t::treatment());

        const auto& gr_view = outer_.grid_range_;

        size_t pos = 0;
        for (size_t i = 0; i < gr_view.n_gridpts(); ++i) {
            if (gr_view.is_regular(i)) {
                rej_len[pos] = (likely(gr_view.check_null(pos, 0)))
                                   ? rej_len_internal(i)
                                   : 0;
                ++pos;
                continue;
            }

            bool internal_called = false;
            size_t rej = 0;
            for (size_t t = 0; t < gr_view.n_tiles(i); ++t, ++pos) {
                bool is_null = gr_view.check_null(pos, 0);
                if (!internal_called && is_null) {
                    rej = rej_len_internal(i);
                    internal_called = true;
                }
                rej_len[pos] = is_null ? rej : 0;
            }
        }

        assert(rej_len.size() == pos);
    }

    using base_t::score;
};

}  // namespace exponential
}  // namespace model
}  // namespace imprint
