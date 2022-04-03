#pragma once
#include <kevlar_bits/model/base.hpp>

namespace kevlar {
namespace experimental {

template <class ValueType>
struct GaussianSimpleModel
    : ModelBase<ValueType>
{
    using value_t = ValueType;
    using base_t = ModelBase<value_t>;

    template <class _GenType,
              class _ValueType,
              class _UIntType,
              class _GridRangeType>
    struct SimGlobalState;

    template <class _GenType,
              class _ValueType,
              class _UIntType,
              class _GridRangeType>
    using sim_global_state_t = SimGlobalState<
        _GenType, _ValueType, _UIntType, 
        _GridRangeType
    >;

    GaussianSimpleModel(
            const Eigen::Ref<const colvec_type<value_t>>& cvs)
        : base_t(cvs)
    {}

    // TODO
    //using upper_bound_state_base_t = UpperBoundStateType;

    template <class _GenType,
              class _ValueType,
              class _UIntType,
              class _GridRangeType>
    auto make_sim_global_state(const _GridRangeType& gr) const
    {
        return sim_global_state_t<
            _GenType, _ValueType, _UIntType,
            _GridRangeType>(*this, gr);
    };

    
    // TODO
    auto make_upper_bound_state() const {};
};

template <class ValueType>
template <class _GenType,
          class _ValueType,
          class _UIntType,
          class _GridRangeType>
struct GaussianSimpleModel<ValueType>::
    SimGlobalState
    : SimGlobalStateBase<
        SimStateBase<_GenType, _ValueType, _UIntType>
      >
{
    struct SimState;

private:
    using sim_state_t = SimState;
    using sim_state_base_t = SimStateBase<
        _GenType, _ValueType, _UIntType>;
    using grid_range_t = _GridRangeType;
    using model_t = GaussianSimpleModel;

    const model_t& model_;
    const grid_range_t& grid_range_;

    // Extra user-defined members only accessed by SimState
    const auto& model() const { return model_; }
    const auto& grid_range() const { return grid_range_; }

public:
    SimGlobalState(
            const model_t& model,
            const grid_range_t& grid_range)
        : model_(model)
        , grid_range_(grid_range)
    {
        // nothing to further compute about grid_range other than
        // the grid-points themselves
    }

    std::unique_ptr<sim_state_base_t> make_sim_state() const override
    {
        return std::make_unique<sim_state_t>(*this);
    }
};

template <class ValueType>
template <class _GenType,
          class _ValueType,
          class _UIntType,
          class _GridRangeType>
struct GaussianSimpleModel<ValueType>::
    SimGlobalState<
        _GenType, _ValueType, _UIntType, _GridRangeType>::
    SimState
    : sim_state_base_t
{
    using base_t = sim_state_base_t;
    using gen_t = typename sim_state_base_t::gen_t;
    using value_t = typename sim_state_base_t::value_t;
    using uint_t = typename sim_state_base_t::uint_t;

    using sim_global_state_t = SimGlobalState;

private:
    const sim_global_state_t& sgs_; // global state
    value_t std_normal_ = std::numeric_limits<value_t>::infinity();    // standard normal r.v.
    std::normal_distribution<value_t> normal_dist_; // standard normal dist

public:
    SimState(const sim_global_state_t& sgs)
        : sgs_(sgs)
        , normal_dist_(0., 1.)
    {}

    void simulate(
            gen_t& gen, 
            Eigen::Ref<colvec_type<uint_t>> rejection_length) override
    {
        // grab global state members
        const auto& model = sgs_.model();
        const auto& gr = sgs_.grid_range();

        // alias
        const auto& cv = model.critical_values();

        // generate a new standard normal
        std_normal_ = normal_dist_(gen);

        size_t pos = 0;
        for (int i = 0; i < gr.n_gridpts(); ++i) {
            auto mu_i = gr.thetas()(0,i);

            // get number of models rejected
            auto it = std::find_if(cv.begin(), cv.end(), 
                        [&](auto t) { return std_normal_ > (t-mu_i); });
            uint_t rej_len = std::distance(it, cv.end());

            for (int j = 0; j < gr.n_tiles(i); ++j, ++pos) {
                rejection_length[pos] = 
                    gr.check_null(pos, 0) 
                    ? rej_len : 0;
            }
        }
    };

    // Score is simply centered Gaussian.
    // Since we internally only store the standard normal,
    // we simply return that and first argument is ignored.
    // Second argument is ignored since we assume only 1 arm.
    value_t score(size_t, size_t) override { return std_normal_; }
};

} // namespace experimental
} // namespace kevlar
