#pragma once
#include <imprint_bits/util/types.hpp>
#include <memory>

namespace imprint {
namespace model {

/*
 * Base class for all model classes.
 */
template <class ValueType>
struct ModelBase {
    using value_t = ValueType;

   private:
    colvec_type<value_t> critical_values_;

   public:
    ModelBase() = default;
    ModelBase(const Eigen::Ref<const colvec_type<value_t>>& cv)
        : critical_values_(cv) {}

    size_t n_models() const { return critical_values_.size(); }
    void critical_values(const Eigen::Ref<const colvec_type<value_t>>& cv) {
        critical_values_ = cv;
    }
    auto& critical_values() { return critical_values_; }
    const auto& critical_values() const { return critical_values_; }
};

/*
 * Base class for all model global state classes.
 * This class contains the interface for all model-specific
 * simulation related global caching and creating simulation states.
 */
template <class ValueType, class UIntType>
struct SimGlobalStateBase {
    struct SimState;

    using interface_t = SimGlobalStateBase;
    using value_t = ValueType;
    using uint_t = UIntType;
    using sim_state_t = SimState;

    virtual ~SimGlobalStateBase(){};

    virtual std::unique_ptr<sim_state_t> make_sim_state(size_t seed) const = 0;
};

/*
 * Base class for all model simulation state classes.
 * This class contains the interface for all model-specific
 * simulation related routines.
 */
template <class ValueType, class UIntType>
struct SimGlobalStateBase<ValueType, UIntType>::SimState {
   private:
    using outer_t = SimGlobalStateBase;

   public:
    using interface_t = SimState;
    using uint_t = typename outer_t::uint_t;
    using value_t = typename outer_t::value_t;

    virtual ~SimState(){};

    /*
     * Simulates model using RNG gen and updates
     * rejection_length with the total number of models falsely rejected.
     * The ith position of rejection_length corresponds to
     * the ith tile in a grid-range.
     */
    virtual void simulate(Eigen::Ref<colvec_type<uint_t>> rejection_length) = 0;

    /*
     * Computes the score of exponential family for parameter at param_idx
     * and grid-point at gridpt_idx.
     */
    virtual void score(size_t gridpt_idx,
                       Eigen::Ref<colvec_type<value_t>> out) const = 0;
};

/*
 * Base class for all model imprint bound state classes.
 * This class contains the interface for all model-specific
 * imprint bound related information.
 * TODO: this interface will need to be further refactored
 * once we start playing around with new imprint bounds.
 */
template <class ValueType>
struct ImprintBoundStateBase {
    using value_t = ValueType;
    using interface_t = ImprintBoundStateBase;

    virtual ~ImprintBoundStateBase(){};

    /*
     * Computes Jacobian of eta evaluated at gridpt given by gridpt_idx
     * and multiplies to v.
     * Eta is the transformation that maps a grid-point to
     * the corresponding natural parameter of the exponential family.
     * The result is stored in out.
     */
    virtual void apply_eta_jacobian(
        size_t gridpt_idx, const Eigen::Ref<const colvec_type<value_t>>& v,
        Eigen::Ref<colvec_type<value_t>> out) = 0;

    /*
     * Computes the covariance (evaluated at gridpt given by gridpt_idx)
     * quadratic form.
     */
    virtual value_t covar_quadform(
        size_t gridpt_idx, const Eigen::Ref<const colvec_type<value_t>>& v) = 0;

    /*
     * Computes an upper bound U(v) of
     *      \sup\limits_{\theta \in \text{tile}} v^\top \nabla^2 f(\theta) v
     * Note that U must be convex.
     * TODO: f is the Type I error function, but possibly generalizable
     * to other functions like bias, MSE, FDR.
     */
    virtual value_t hessian_quadform_bound(
        size_t gridpt_idx, size_t tile_idx,
        const Eigen::Ref<const colvec_type<value_t>>& v) = 0;

    /*
     * Returns the number of natural parameters.
     */
    virtual size_t n_natural_params() const = 0;
};

}  // namespace model
}  // namespace imprint
