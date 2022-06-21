#pragma once
#include <algorithm>
#include <imprint_bits/model/base.hpp>
#include <imprint_bits/model/binomial/common/fixed_n_default.hpp>
#include <imprint_bits/model/fixed_single_arm_size.hpp>
#include <imprint_bits/stat/unpaired_test.hpp>
#include <imprint_bits/util/macros.hpp>

namespace imprint {
namespace model {
namespace binomial {

template <class ValueType>
struct SimpleSelection : FixedSingleArmSize, ModelBase<ValueType> {
    using arm_base_t = FixedSingleArmSize;
    using base_t = ModelBase<ValueType>;
    using typename base_t::value_t;

   private:
    const size_t n_phase2_samples_;

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
        ImprintBoundStateFixedNDefault<_GridRangeType>;

    SimpleSelection(size_t n_arms, size_t n_arm_samples,
                    size_t n_phase2_samples,
                    const Eigen::Ref<const colvec_type<value_t>>& cv)
        : arm_base_t(n_arms, n_arm_samples),
          base_t(),
          n_phase2_samples_(n_phase2_samples) {
        assert(n_phase2_samples <= n_arm_samples);
        critical_values(cv);
    }

    using arm_base_t::n_arm_samples;
    using arm_base_t::n_arms;

    IMPRINT_STRONG_INLINE
    constexpr size_t n_phase2_samples() const { return n_phase2_samples_; }

    using base_t::critical_values;
    void critical_values(const Eigen::Ref<const colvec_type<value_t>>& cv) {
        auto& cv_ = base_t::critical_values();
        cv_ = cv;
        std::sort(cv_.data(), cv_.data() + cv_.size(), std::greater<value_t>());
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
};

template <class ValueType>
template <class _GenType, class _ValueType, class _UIntType,
          class _GridRangeType>
struct SimpleSelection<ValueType>::SimGlobalState
    : SimGlobalStateFixedNDefault<_GenType, _ValueType, _UIntType,
                                  _GridRangeType> {
    struct SimState;

    using base_t = SimGlobalStateFixedNDefault<_GenType, _ValueType, _UIntType,
                                               _GridRangeType>;
    using typename base_t::gen_t;
    using typename base_t::grid_range_t;
    using typename base_t::interface_t;
    using typename base_t::uint_t;
    using typename base_t::value_t;

    using sim_state_t = SimState;

   private:
    using model_t = SimpleSelection;
    const model_t& model_;

   public:
    SimGlobalState(const model_t& model, const grid_range_t& grid_range)
        : base_t(model.n_arm_samples(), grid_range), model_(model) {}

    std::unique_ptr<typename interface_t::sim_state_t> make_sim_state(
        size_t seed) const override {
        return std::make_unique<sim_state_t>(*this, seed);
    }
};

template <class ValueType>
template <class _GenType, class _ValueType, class _UIntType,
          class _GridRangeType>
struct SimpleSelection<ValueType>::SimGlobalState<
    _GenType, _ValueType, _UIntType, _GridRangeType>::SimState
    : base_t::sim_state_t {
   private:
    using outer_t = SimGlobalState;

   public:
    using base_t = typename outer_t::base_t::sim_state_t;
    using typename base_t::interface_t;

   private:
    const outer_t& outer_;
    colvec_type<uint_t>
        phase2_counts_;  // sufficient statistic table only looking at phase 2
                         // and treatment arms phase2_counts_(i,j) = phase2 suff
                         // stat at unique prob i at arm j.

    /*
     * Creates a view of jth arm Phase II counts.
     * Note that 1 <= j < n_arms.
     */
    template <bool do_const>
    IMPRINT_STRONG_INLINE auto phase2_counts_arm(size_t j) const {
        using vec_t = std::conditional_t<do_const, const colvec_type<uint_t>,
                                         colvec_type<uint_t>>;
        auto& ph2_casted = const_cast<vec_t&>(phase2_counts_);
        const auto& sgs = outer_;
        return Eigen::Map<vec_t>(
            ph2_casted.data() + sgs.stride(j) - sgs.stride(1),
            sgs.stride(j + 1) - sgs.stride(j));
    }

   protected:
    IMPRINT_STRONG_INLINE
    auto phase2_counts_arm(size_t j) const {
        return phase2_counts_arm<true>(j);
    }

    IMPRINT_STRONG_INLINE
    auto phase2_counts_arm(size_t j) { return phase2_counts_arm<false>(j); }

    /*
     * Generates sufficient statistic for each arm under all possible grid
     * points.
     * Note that this technically does extra computations than necessary,
     * but benchmarking shows it makes no difference from the more optimized
     * one. For simplicity and readability, we choose this version.
     */
    void generate_sufficient_stats() {
        // generate sufficient stats only for phase II
        const auto& sgs = outer_;
        const auto& model = sgs.model_;

        auto& uniform_randoms = base_t::uniform_randoms();
        const auto n_params = uniform_randoms.cols();

        // grab the block of uniforms associated with Phase II/III for
        // treatments.
        const size_t phase2_size = model.n_phase2_samples();
        auto phase2_unif =
            uniform_randoms.block(0, 1, phase2_size, n_params - 1);

        // sort each column of each block.
        sort_cols(phase2_unif);

        const auto phase2_counts_size = sgs.stride(n_params) - sgs.stride(1);
        phase2_counts_.resize(phase2_counts_size);

        for (size_t i = 1; i < n_params; ++i) {
            auto phase2_counts_i = phase2_counts_arm(i);
            accum_count(phase2_unif.col(i - 1), sgs.probs_unique_arm(i),
                        phase2_counts_i);
        }

        // generate full sufficient stats
        base_t::generate_sufficient_stats();
    }

    template <class BitsType>
    IMPRINT_STRONG_INLINE auto phase_III_internal(
        size_t a_star, const BitsType& bits_i) const {
        const auto& sgs = outer_;
        const auto& model = sgs.model_;

        auto n = model.n_arm_samples();
        auto ss_astar = base_t::sufficient_stats_arm(a_star);
        auto ss_0 = base_t::sufficient_stats_arm(0);

        // unpaired z-test with binomial approximation
        int x_s = static_cast<int64_t>(ss_astar(bits_i[a_star]));
        int x_0 = static_cast<int64_t>(ss_0(bits_i[0]));
        auto z = stat::UnpairedTest<value_t>::binom_stat(x_s, x_0, n);

        const auto& cv = model.critical_values();
        int i = 0;
        for (; i < cv.size(); ++i) {
            if (z > cv[i]) break;
        }
        return outer_.model_.n_models() - i;
    };

   public:
    SimState(const outer_t& sgs, size_t seed)
        : base_t(sgs, seed), outer_(sgs) {}

    void simulate(Eigen::Ref<colvec_type<uint_t>> rej_len) override {
        // sample binomial data for the whole grid-range
        base_t::generate_data();
        generate_sufficient_stats();

        const auto& sgs = outer_;
        const auto& bits = sgs.bits();
        const auto& gr_view = sgs.grid_range();

        size_t pos = 0;
        for (int i = 0; i < gr_view.n_gridpts(); ++i) {
            const auto bits_i = bits.col(i);

            // Phase II
            int a_star =
                -1;  // selected arm with highest Phase II response count.
            int max_count = -1;  // maximum Phase II response count.
            for (int j = 1; j < bits_i.size(); ++j) {
                int prev_count = max_count;
                auto phase2_counts_v = phase2_counts_arm(j);
                max_count =
                    std::max(static_cast<int64_t>(prev_count),
                             static_cast<int64_t>(phase2_counts_v(bits_i[j])));
                a_star = (max_count != prev_count) ? j : a_star;
            }

            // Phase III

            size_t rej = 0;

            // if current gridpt is regular, do an optimized routine.
            if (gr_view.is_regular(i)) {
                if (gr_view.check_null(pos, a_star - 1)) {
                    rej = phase_III_internal(a_star, bits_i);
                }
                rej_len[pos] = rej;
                ++pos;
                continue;
            }

            // else, do a slightly different routine:
            // compute the phase3 test statistic first and loop through each
            // tile to check if it's a false rejection.
            bool rej_computed = false;
            const auto n_ts = gr_view.n_tiles(i);
            for (size_t n_t = 0; n_t < n_ts; ++n_t, ++pos) {
                bool is_null = gr_view.check_null(pos, a_star - 1);
                if (!rej_computed && is_null) {
                    rej = phase_III_internal(a_star, bits_i);
                    rej_computed = true;
                }
                rej_len[pos] = is_null ? rej : 0;
            }
        }
        assert(rej_len.size() == pos);
    }

    using base_t::score;
};

}  // namespace binomial
}  // namespace model
}  // namespace imprint
