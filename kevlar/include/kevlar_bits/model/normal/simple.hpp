#pragma once
#include <kevlar_bits/distribution/normal.hpp>
#include <kevlar_bits/model/base.hpp>
#include <kevlar_bits/model/fixed_single_arm_size.hpp>
#include <limits>

namespace kevlar {
namespace model {
namespace normal {

template <class ValueType>
struct Simple : FixedSingleArmSize, ModelBase<ValueType> {
    using base_t = ModelBase<ValueType>;
    using typename base_t::value_t;

   private:
    using arm_t = FixedSingleArmSize;

    KEVLAR_STRONG_INLINE
    constexpr auto n_params() const { return 1; }

   public:
    template <class _GenType, class _ValueType, class _UIntType,
              class _GridRangeType>
    struct SimGlobalState;

    template <class _ValueType, class _TileType>
    struct KevlarBoundState;

    Simple(const Eigen::Ref<const colvec_type<value_t>>& cvs)
        : arm_t(1, 1), base_t(cvs) {}

    template <class _GenType, class _ValueType, class _UIntType,
              class _GridRangeType>
    auto make_sim_global_state(const _GridRangeType& gr) const {
        return SimGlobalState<_GenType, _ValueType, _UIntType, _GridRangeType>(
            *this, gr);
    };

    template <class _ValueType, class _TileType>
    auto make_kevlar_bound_state() const {
        return KevlarBoundState<_ValueType, _TileType>(*this);
    };
};

template <class ValueType>
template <class _GenType, class _ValueType, class _UIntType,
          class _GridRangeType>
struct Simple<ValueType>::SimGlobalState
    : SimGlobalStateBase<_GenType, _ValueType, _UIntType> {
    struct SimState;

    using base_t = SimGlobalStateBase<_GenType, _ValueType, _UIntType>;
    using interface_t = base_t;
    using typename base_t::gen_t;
    using typename base_t::uint_t;
    using typename base_t::value_t;
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

    std::unique_ptr<typename interface_t::sim_state_t> make_sim_state()
        const override {
        return std::make_unique<sim_state_t>(*this);
    }
};

template <class ValueType>
template <class _GenType, class _ValueType, class _UIntType,
          class _GridRangeType>
struct Simple<ValueType>::SimGlobalState<_GenType, _ValueType, _UIntType,
                                         _GridRangeType>::SimState
    : SimGlobalState::interface_t::sim_state_t {
    using outer_t = SimGlobalState;
    using base_t = typename outer_t::interface_t::sim_state_t;
    using interface_t = typename base_t::interface_t;

   private:
    using normal_t = distribution::Normal<value_t>;

    const outer_t& outer_;  // global state
    normal_t normal_;       // standard normal object
    value_t std_normal_ =
        std::numeric_limits<value_t>::infinity();  // standard normal r.v.

   public:
    SimState(const outer_t& outer) : outer_(outer), normal_(0., 1.) {}

    void simulate(gen_t& gen,
                  Eigen::Ref<colvec_type<uint_t>> rejection_length) override {
        // grab global state members
        const auto& model = outer_.model();
        const auto& gr = outer_.grid_range();

        // alias
        const auto& cv = model.critical_values();

        // generate a new standard normal
        std_normal_ = normal_.sample(gen);

        size_t pos = 0;
        for (int i = 0; i < gr.n_gridpts(); ++i) {
            auto mu_i = gr.thetas()(0, i);

            // get number of models rejected
            auto it = std::find_if(cv.begin(), cv.end(), [&](auto t) {
                return std_normal_ > (t - mu_i);
            });
            uint_t rej_len = std::distance(it, cv.end());

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
template <class _ValueType, class _TileType>
struct Simple<ValueType>::KevlarBoundState
    : KevlarBoundStateBase<_ValueType, _TileType> {
    using base_t = KevlarBoundStateBase<_ValueType, _TileType>;
    using typename base_t::interface_t;
    using typename base_t::tile_t;
    using typename base_t::value_t;

    void apply_eta_jacobian(const Eigen::Ref<const colvec_type<value_t>>&,
                            const Eigen::Ref<const colvec_type<value_t>>& v,
                            Eigen::Ref<colvec_type<value_t>> out) override {
        out = v;
    }

    value_t covar_quadform(
        const Eigen::Ref<const colvec_type<value_t>>&,
        const Eigen::Ref<const colvec_type<value_t>>& v) override {
        return v.squaredNorm();
    }

    value_t hessian_quadform_bound(
        const tile_t&,
        const Eigen::Ref<const colvec_type<value_t>>& v) override {
        return v.squaredNorm();
    }

    size_t n_natural_params() const override { return 1; }
};

}  // namespace normal
}  // namespace model
}  // namespace kevlar
