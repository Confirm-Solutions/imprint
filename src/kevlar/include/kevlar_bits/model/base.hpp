#pragma once
#include <kevlar_bits/util/types.hpp>
#include <memory>
#include <random>  // TODO: should generalize

namespace kevlar {

template <class ValueType, class UIntType, class GridRangeType>
struct ModelStateBase;

/*
 * Base class for all model classes.
 */
template <class ValueType, class UIntType, class GridRangeType>
struct ModelBase {
    using value_t = ValueType;
    using uint_t = UIntType;
    using gr_t = GridRangeType;
    using model_state_t = ModelStateBase<value_t, uint_t, gr_t>;

    virtual ~ModelBase() = default;
    virtual value_t cov_quad(
        size_t, const Eigen::Ref<const colvec_type<value_t>>&) const = 0;
    virtual value_t max_cov_quad(
        size_t, const Eigen::Ref<const colvec_type<value_t>>&) const = 0;
    virtual uint_t n_models() const = 0;
    virtual std::unique_ptr<model_state_t> make_state() const = 0;

    void set_grid_range(const gr_t& grid_range) {
        grid_range_view_ = &grid_range;
    }
    const gr_t& grid_range() const { return *grid_range_view_; }

   private:
    const gr_t* grid_range_view_ = nullptr;  // viewer of current grid range
};

/*
 * Base class for all model state classes.
 */
template <class ValueType, class UIntType, class GridRangeType>
struct ModelStateBase {
    using value_t = ValueType;
    using uint_t = UIntType;
    using gr_t = GridRangeType;
    using model_base_t = ModelBase<value_t, uint_t, gr_t>;

    ModelStateBase(const model_base_t& mb) : mb_{&mb} {}

    virtual ~ModelStateBase() = default;
    virtual void gen_rng(std::mt19937&) = 0;  // TODO: generalize mt19937
    virtual void gen_suff_stat() = 0;
    virtual void rej_len(Eigen::Ref<colvec_type<uint_t>>) = 0;
    virtual value_t grad(uint_t, uint_t) = 0;

    const gr_t& grid_range() { return mb_->grid_range(); }

   private:
    const model_base_t* mb_;
};

/*
 * Base class for all control + k treatment designs.
 */
struct ControlkTreatmentBase {
    /*
     * @param   n_arms      number of arms (including control).
     * @param   ph2_size    phase II number of patients in each arm.
     * @param   n_samples   number of total patients in each arm (including
     * phase II) for phase II and phase III.
     */
    ControlkTreatmentBase(size_t n_arms, size_t ph2_size, size_t n_samples)
        : n_arms_(n_arms), ph2_size_(ph2_size), n_samples_(n_samples) {}

    constexpr size_t n_arms() const { return n_arms_; }
    constexpr size_t ph2_size() const { return ph2_size_; }
    constexpr size_t n_samples() const { return n_samples_; }

    /* Helper static interface */
    template <class GenType, class UnifType, class OutType>
    static void uniform(size_t m, size_t n, GenType&& gen, UnifType&& unif,
                        OutType&& out) {
        out.array() =
            out.NullaryExpr(m, n, [&](auto, auto) { return unif(gen); });
    }

   protected:
    size_t n_arms_;
    size_t ph2_size_;
    size_t n_samples_;
};

}  // namespace kevlar
