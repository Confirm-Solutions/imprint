#pragma once
#include <Eigen/Core>
#include <kevlar_bits/util/math.hpp>

namespace kevlar {

/*
 * This class is a generic object for capturing all the logic of constructing an upper bound.
 * Given an exponential family model object satisfying a certain interface (see member functions below),
 * it stores all necessary components of the upper bound (zeroth/first order term and zeroth/first/second order upper bound corrections).
 *
 * @param   ValueType       underlying value type (usually just double).
 */
template <class ValueType>
struct UpperBound
{
private:
    using value_t = ValueType;

    /*
     * Upper bound correction for 0th order term (just naive plug-in confidence bound):
     *
     * z_{1-\delta / 2} * \sqrt{(\delta_0 * (1-\delta_0) / n)}
     *
     * @param   delta       confidence for provable upper bound.
     */
    auto upper_bd_constant(value_t delta) const
    {
        auto width = qnorm(1.-delta/2.);
        return (width * (upper_bd_.array() * (1. - upper_bd_.array()) / n_).sqrt()).matrix();
    }

    /*
     * Computes the upper bound for gradient and hessian term (separately).
     * Applies functor f to these quantities under each grid point.
     *
     * @param   model                   underlying model (design).
     * @param   mean_idxer_range        range object representing the grid points.
     * @param   delta                   confidence for provable upper bound.
     * @param   grid_radius             radius of the grid in Theta space.
     * @param   f                       functor to apply on gradient/hessian upper bounds under each grid point.
     */
    template <class ModelType
            , class MeanIdxerRangeType
            , class FType>
    void upper_bd_grad_hess(
            const ModelType& model,
            const MeanIdxerRangeType& mean_idxer_range,
            value_t delta,
            value_t grid_radius,
            FType f) const 
    {
        auto idxer_begin = mean_idxer_range.begin();

        for (Eigen::Index i = 0; i < upper_bd_.cols(); ++i, ++idxer_begin) {
            
            auto& idxer = *idxer_begin;

            // compute grad upper bound
            value_t var = model.tr_cov(idxer);
                
            var *= (2./delta - 1.) / n_;
            value_t grad_bd = grid_radius * std::sqrt(var);

            // compute hessian upper bound
            value_t hess_bd = model.tr_max_cov(idxer);
            hess_bd *= (grid_radius * grid_radius) / 2.;
            f(i, grad_bd, hess_bd);
        }
    }

    constexpr auto n_arms() const { return grad_buff_.size() / upper_bd_.size(); }

public:

    /*
     * Pools other upper bound objects into the current object.
     * Must be in state Createable to be a valid call.
     *
     * @param   other       other UpperBound object to pool into current object.
     */
    void pool(const UpperBound& other)
    {
        upper_bd_ += other.upper_bd_;
        grad_buff_ += other.grad_buff_;
        n_ += other.n_;
    }

    /*
     * Creates a full upper bound estimate.
     * Must be in state Createable to be a valid call.
     * After the call, the state is not in Createable.
     *
     * @param   mean_idxer_range     range object for the grid points.
     * @param   delta                confidence of provable upper bound.
     * @param   grid_radius          radius of the grid in the natural parameter space.
     */
    template <class ModelType, class MeanIdxerRangeType>
    void create(const ModelType& model,
                const MeanIdxerRangeType& mean_idxer_range, 
                value_t delta,
                value_t grid_radius)
    {
        // divide to get true averages
        upper_bd_ /= n_;
        grad_buff_ /= n_;

        // add upper bound for constant term
        upper_bd_ += upper_bd_constant(delta);

        // add epsilon * ||grad term||_L^1
        const auto slice_size = upper_bd_.size();
        auto slice_offset = 0;
        size_t n_arms = this->n_arms();
        for (size_t k = 0; k < n_arms; ++k, slice_offset += slice_size) {
            Eigen::Map<mat_t> grad_k(
                    grad_buff_.data() + slice_offset,
                    upper_bd_.rows(),
                    upper_bd_.cols());
            upper_bd_.array() += grid_radius * grad_k.array().abs();
        }

        // add upper bound for gradient term and hessian term
        upper_bd_grad_hess(
            model, mean_idxer_range, delta, grid_radius,
            [&](Eigen::Index i, auto grad_bd, auto hess_bd) {
                upper_bd_.col(i).array() += grad_bd + hess_bd;
            });
    }

    /*
     * Computes a hint for what the batch size of parameters should be.
     * Given the number of thresholds (thr_vec_size) to consider,
     * we estimate this from a pre-fit GLM model.
     */
    constexpr auto p_batch_size_hint(size_t thr_vec_size) const 
    {
        // Heuristic for guessing how much can fit in the cache:
        // Performance seems pretty good according to this formula,
        // though it may depend on the machine.
        size_t out = 7430 * std::exp(-0.035717058454485931 * thr_vec_size);
        return out;
    }

private:
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;

    // This matrix will be used to store the final upper bound.
    // To save memory, it will be used to store the rejection proportions during update.
    mat_t upper_bd_;
    vec_t grad_buff_;
    size_t n_ = 0;              // number of updates
    bool serialized_ = false;   // true iff serialize() has been called.
};

} // namespace kevlar
