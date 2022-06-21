#pragma once
#include <imprint_bits/distribution/normal.hpp>
#include <imprint_bits/model/base.hpp>
#include <imprint_bits/model/fixed_single_arm_size.hpp>
#include <limits>

namespace imprint {
namespace model {
namespace normal {

template <class ValueType>
struct Simple : FixedSingleArmSize, ModelBase<ValueType> {
    using arm_t = FixedSingleArmSize;
    using base_t = ModelBase<ValueType>;
    using typename base_t::value_t;

   private:
    IMPRINT_STRONG_INLINE
    constexpr auto n_params() const { return 1; }

   public:
    template <class _GenType, class _ValueType, class _UIntType,
              class _GridRangeType>
    struct SimGlobalState;

    template <class _GridRangeType>
    struct ImprintBoundState;

    template <class _GenType, class _ValueType, class _UIntType,
              class _GridRangeType>
    using sim_global_state_t =
        SimGlobalState<_GenType, _ValueType, _UIntType, _GridRangeType>;

    template <class _GridRangeType>
    using imprint_bound_state_t = ImprintBoundState<_GridRangeType>;

    Simple(const Eigen::Ref<const colvec_type<value_t>>& cvs)
        : arm_t(1, 1), base_t(cvs) {}

    template <class _GenType, class _ValueType, class _UIntType,
              class _GridRangeType>
    auto make_sim_global_state(const _GridRangeType& gr) const {
        return sim_global_state_t<_GenType, _ValueType, _UIntType,
                                  _GridRangeType>(*this, gr);
    };

    // grid range is not used, but we keep it as a parameter for consistency.
    template <class _GridRangeType>
    auto make_imprint_bound_state(const _GridRangeType&) const {
        return imprint_bound_state_t<_GridRangeType>();
    };
};

template <class ValueType>
template <class _GenType, class _ValueType, class _UIntType,
          class _GridRangeType>
struct Simple<ValueType>::SimGlobalState
    : SimGlobalStateBase<_ValueType, _UIntType> {
    struct SimState;

    using base_t = SimGlobalStateBase<_ValueType, _UIntType>;
    using typename base_t::interface_t;
    using typename base_t::uint_t;
    using typename base_t::value_t;
    using gen_t = _GenType;
    using grid_range_t = _GridRangeType;

    using sim_state_t = SimState;

   private:
    using model_t = Simple;

    const model_t& model_;
    const grid_range_t& grid_range_;

    // Extra user-defined members only accessed by SimState
    const auto& model() const { return model_; }
    const auto& grid_range() const { return grid_range_; }

   public:
    SimGlobalState(const model_t& model, const grid_range_t& grid_range)
        : model_(model), grid_range_(grid_range) {
        // nothing to further compute about grid_range other than
        // the grid-points themselves
    }

    std::unique_ptr<typename interface_t::sim_state_t> make_sim_state(
        size_t seed) const override {
        return std::make_unique<sim_state_t>(*this, seed);
    }
};

template <class ValueType>
template <class _GenType, class _ValueType, class _UIntType,
          class _GridRangeType>
struct Simple<ValueType>::SimGlobalState<_GenType, _ValueType, _UIntType,
                                         _GridRangeType>::SimState
    : SimGlobalState::interface_t::sim_state_t {
   private:
    using outer_t = SimGlobalState;

   public:
    using base_t = typename outer_t::interface_t::sim_state_t;
    using interface_t = typename base_t::interface_t;

   private:
    using normal_t = distribution::Normal<value_t>;

    const outer_t& outer_;  // global state
    normal_t normal_;       // standard normal object
    value_t std_normal_ =
        std::numeric_limits<value_t>::infinity();  // standard normal r.v.
    gen_t gen_;

   public:
    SimState(const outer_t& outer, size_t seed)
        : outer_(outer), normal_(0., 1.), gen_(seed) {}

    void simulate(Eigen::Ref<colvec_type<uint_t>> rejection_length) override {
        // grab global state members
        const auto& model = outer_.model();
        const auto& gr = outer_.grid_range();

        // alias
        const auto& cv = model.critical_values();

        // generate a new standard normal
        std_normal_ = normal_.sample(gen_);

        size_t pos = 0;
        for (int i = 0; i < gr.n_gridpts(); ++i) {
            auto mu_i = gr.thetas()(0, i);

            // get number of models rejected
            int j = 0;
            for (; j < cv.size(); ++j) {
                if ((std_normal_ + mu_i) > cv[j]) {
                    break;
                }
            }
            uint_t rej_len = cv.size() - j;

            for (int j = 0; j < gr.n_tiles(i); ++j, ++pos) {
                rejection_length[pos] = gr.check_null(pos, 0) ? rej_len : 0;
            }
        }

        assert(rejection_length.size() == pos);
    };

    // Score is simply centered Normal.
    // Since we internally only store the standard normal,
    // we simply return that and first argument is ignored.
    // Second argument is ignored since we assume only 1 arm.
    void score(size_t, Eigen::Ref<colvec_type<value_t>> out) const override {
        assert(out.size() == outer_.model_.n_params());
        out[0] = std_normal_;
    }
};

template <class ValueType>
template <class _GridRangeType>
struct Simple<ValueType>::ImprintBoundState
    : ImprintBoundStateBase<typename _GridRangeType::value_t> {
    using grid_range_t = _GridRangeType;
    using base_t = ImprintBoundStateBase<typename grid_range_t::value_t>;
    using typename base_t::interface_t;
    using typename base_t::value_t;

    void apply_eta_jacobian(size_t,
                            const Eigen::Ref<const colvec_type<value_t>>& v,
                            Eigen::Ref<colvec_type<value_t>> out) override {
        out = v;
    }

    value_t covar_quadform(
        size_t, const Eigen::Ref<const colvec_type<value_t>>& v) override {
        return v.squaredNorm();
    }

    value_t hessian_quadform_bound(
        size_t, size_t,
        const Eigen::Ref<const colvec_type<value_t>>& v) override {
        return v.squaredNorm();
    }

    size_t n_natural_params() const override { return 1; }
};

}  // namespace normal
}  // namespace model
}  // namespace imprint
