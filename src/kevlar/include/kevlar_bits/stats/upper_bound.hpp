#pragma once
#include <Eigen/Core>
#include <kevlar_bits/util/math.hpp>
#include <kevlar_bits/util/types.hpp>

namespace kevlar {

/*
 * This class encapsulates the logic of constructing an upper bound.
 * It stores all necessary components of the upper bound.
 *
 * @param   ValueType       underlying value type (usually double).
 */
template <class ValueType>
struct UpperBound
{
    using value_t = ValueType;

    /*
     * Creates and stores the components to the upper bound estimate.
     *
     * @param   model               ModelBase-like object.
     * @param   is_o                InterSum-like object.
     *                              Must be an object updated from simulating under
     *                              model attached with grid_range.
     * @param   grid_range          GridRange-like object.
     *                              Must be the same grid range that is_o
     *                              was updated with.
     * @param   delta               confidence of provable upper bound.
     * @param   delta_prop_0to1     proportion of delta to put 
     *                              into 0th order upper bound.
     */
    template <class ModelType
            , class InterSumType
            , class GridRangeType>
    void create(const ModelType& model,
                const InterSumType& is_o, 
                const GridRangeType& grid_range,
                value_t delta,
                value_t delta_prop_0to1 = 0.5)
    {
        // some aliases
        auto n_models = is_o.n_models();
        auto n_gridpts = is_o.n_gridpts();
        auto n_params = is_o.n_params();
        auto& sim_sizes = grid_range.get_sim_sizes();

        // populate 0th order and upper bound
        delta_0_.resize(n_models, n_gridpts);
        delta_0_u_.resize(delta_0_.rows(), delta_0_.cols());
        auto width = qnorm(1.-delta * delta_prop_0to1);
        auto& typeIsum = is_o.type_I_sum();
        for (int j = 0; j < delta_0_u_.cols(); ++j) {
            auto delta_0_j = delta_0_.col(j);
            delta_0_j = typeIsum.col(j).template cast<value_t>() /
                sim_sizes[j];
            delta_0_u_.col(j) = (width / std::sqrt(sim_sizes[j])) * 
                (delta_0_j.array() * (1.0-delta_0_j.array())).sqrt();
        }

        // populate 1st order
        const auto slice_size = n_models * n_gridpts;
        size_t slice_offset = 0;
        delta_1_.setZero(n_models, n_gridpts);
        auto& radii = grid_range.get_radii();
        for (size_t k = 0; k < n_params; ++k, slice_offset += slice_size) {
            Eigen::Map<const mat_type<value_t> > grad_k(
                    is_o.grad_sum().data() + slice_offset,
                    n_models,
                    n_gridpts);
            for (size_t j = 0; j < n_gridpts; ++j) {
                delta_1_.col(j).array() += 
                    (radii(k,j) / sim_sizes[j]) * 
                    grad_k.col(j).array().abs();
            }
        }

        // populate 1st/2nd order upper bound together
        value_t correction = std::sqrt(1./((1.0-delta_prop_0to1)*delta) - 1.);
        delta_1_u_.resize(n_gridpts);
        delta_2_u_.resize(n_gridpts);
        for (int j = 0; j < delta_1_u_.size(); ++j) {
            value_t var = 0.0;
            value_t hess_bd = 0.0;
            for (size_t k = 0; k < n_params; ++k) {
                auto radii_sq = radii(k,j) * radii(k,j);
                var += model.cov(j,k) * radii_sq;
                hess_bd += model.max_cov(j,k) * radii_sq;
            }
            var /= sim_sizes[j];
            delta_1_u_[j] = var;
            delta_2_u_[j] = hess_bd * 0.5;
        }
        delta_1_u_.array() = delta_1_u_.array().sqrt() * correction;
    }

    mat_type<value_t>& get_delta_0() { return delta_0_; }
    mat_type<value_t>& get_delta_0_u() { return delta_0_u_; }
    mat_type<value_t>& get_delta_1() { return delta_1_; }
    colvec_type<value_t>& get_delta_1_u() { return delta_1_u_; }
    colvec_type<value_t>& get_delta_2_u() { return delta_2_u_; }
    const mat_type<value_t>& get_delta_0() const { return delta_0_; }
    const mat_type<value_t>& get_delta_0_u() const { return delta_0_u_; }
    const mat_type<value_t>& get_delta_1() const { return delta_1_; }
    const colvec_type<value_t>& get_delta_1_u() const { return delta_1_u_; }
    const colvec_type<value_t>& get_delta_2_u() const { return delta_2_u_; }

private:

    // Components that make up an upper bound.
    mat_type<value_t> delta_0_;         // 0th order (n_models x n_gridpts)
    mat_type<value_t> delta_0_u_;       // 0th order upper bound (n_models x n_gridpts)
    mat_type<value_t> delta_1_;         // 1st order (n_models x n_gridpts)
    colvec_type<value_t> delta_1_u_;    // 1st order upper bound (n_gridpts)
    colvec_type<value_t> delta_2_u_;    // 2nd order upper bound (n_gridpts)
};

} // namespace kevlar
