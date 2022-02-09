#pragma once
#include <kevlar_bits/util/types.hpp>

namespace kevlar {

/*
 * Base class for all model state classes.
 */
template <class ValueType, class IntType>
struct ModelStateBase
{
    using value_t = ValueType;
    using int_t = IntType;

    virtual void get_rej_len(Eigen::Ref<colvec_type<int_t> >) const =0;
    virtual void get_grad(Eigen::Ref<colvec_type<value_t> >) const =0;
};

/*
 * Base class for all control + k treatment designs.
 */
struct ControlkTreatmentBase
{
    /*
     * @param   n_arms      number of arms (including control).
     * @param   ph2_size    phase II number of patients in each arm.
     * @param   n_samples   number of total patients in each arm (including phase II) for phase II and phase III.
     */
    ControlkTreatmentBase(
            size_t n_arms,
            size_t ph2_size,
            size_t n_samples
            )
        : n_arms_(n_arms)
        , ph2_size_(ph2_size)
        , n_samples_(n_samples)
    {}

    size_t n_samples() const { return n_samples_; }
    size_t n_arms() const { return n_arms_; }

    /* Helper static interface */
    template <class FloatType, class GenType, class OutType>
    static void uniform(
            FloatType min, 
            FloatType max, 
            GenType&& gen, 
            OutType&& out, 
            size_t m, 
            size_t n) 
    {
        out.array() = 
            (out.Random(m, n).array() + 1) 
            * (static_cast<FloatType>(0.5) * (max-min)) 
            + min;
    }

protected:
    size_t n_arms_;
    size_t ph2_size_;
    size_t n_samples_;
};

} // namespace kevlar
