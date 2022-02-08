#pragma once
#include <kevlar_bits/util/types.hpp>

namespace kevlar {

template <class ValueType>
struct InterSum
{
    using value_t = ValueType;

    InterSum() =default;
    InterSum(
        size_t n_models,
        size_t n_gridpts,
        size_t n_params)
        : type_I_sum_(n_models, n_gridpts)
        , grad_sum_(n_models * n_gridpts * n_params)
        , rej_len_(n_gridpts)
    {
        type_I_sum_.setZero();
        grad_sum_.setZero();
        rej_len_.setZero();
    }
    
    /*
     * Updates type_I_sum and grad_sum estimates based on current model state.
     * Increments type_I_sum by rejection indicators.
     * Increments grad_sum by rejection indicators * (T - \nabla A).
     *
     * @param   state               underlying model state.
     *                              Assumes that the sequence of models considered
     *                              is ordered in ascending order, in the sense that
     *                              if ith model rejects then jth model rejects for all j >= i.
     */
    template <class ModelStateType>
    void update(ModelStateType&& state)
    {
        // increment accumulation counter
        ++n_;

        // get index of first model that rejects for each gridpt
        state.get_rej_len(rej_len_);

        auto n_gridpts = type_I_sum_.cols();

        // update type_I_sum
        for (int j = 0; j < n_gridpts; ++j) {
            type_I_sum_.col(j).tail(rej_len_[j]).array() += 1;
        }
        
        // update gradient for each dimension
        const auto slice_size = type_I_sum_.size();
        auto slice_offset = 0;
        auto n_params = this->n_params();

        // add (T - nabla_eta A(m)) for each threshold where we have rejection.
        // where T is the sufficient statistic for arm k under mean m,
        // nabla_eta A(m) is the gradient under the natural parameter eta of the log-partition function for arm k evaluated at mean m.
        for (int k = 0; k < n_params; ++k, slice_offset += slice_size) {
            Eigen::Map<mat_type<value_t> > grad_k_cache(
                    grad_sum_.data() + slice_offset, 
                    type_I_sum_.rows(),
                    type_I_sum_.cols());
            for (int j = 0; j < n_gridpts; ++j) {
                auto grad_k_j = grad_k_cache.col(j);
                grad_k_j.tail(rej_len_[j]).array() += state.grad_lr(k, j);
            }
        }
    }

    /*
     * Reset the size of internal data structures corresponding
     * to the new configuration n_models, n_gridpts, n_params, n_acc.
     * The first three parameters must be positive.
     *
     * @param   n_models        number of models.
     * @param   n_gridpts       number of gridpts.
     * @param   n_params        number of parameters.
     * @param   n_acc           number of accumulations.
     */
    void reset(
        size_t n_models,
        size_t n_gridpts,
        size_t n_params,
        size_t n_acc=0)
    {
        type_I_sum_.setZero(n_models, n_gridpts);
        grad_sum_.setZero(n_models * n_gridpts * n_params);
        rej_len_.setZero(n_gridpts);
        n_ = n_acc;
    }

    const auto& type_I_sum() const { return type_I_sum_; }
    const auto& grad_sum() const { return grad_sum_; }
    auto n_accum() const { return n_; }

private:
    constexpr auto n_params() const {
        return grad_sum_.size() / type_I_sum_.size();
    }     

    mat_type<uint32_t> type_I_sum_; // Type I error sums.
                                    // type_I_sum_(i,j) = rejection accumulation for model i at gridpt j.
    colvec_type<value_t> grad_sum_; // gradient sums.
                                    // grad_sum_(i,j,k) = partial deriv accumulation w.r.t. param k for model i at gridpt j.
    size_t n_=0;                    // number of accumulations
    colvec_type<uint32_t> rej_len_; // number of models that rejects for each gridpt
};

} // namespace kevlar
