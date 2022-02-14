#pragma once
#include <kevlar_bits/util/types.hpp>
#include <kevlar_bits/util/macros.hpp>

namespace kevlar {

template <class ValueType
        , class UIntType>
struct InterSum
{
    using value_t = ValueType;
    using uint_t = UIntType;

    InterSum() =default;
    InterSum(
        size_t n_models,
        size_t n_gridpts,
        size_t n_params)
        : type_I_sum_(n_models, n_gridpts)
        , grad_sum_(n_models * n_gridpts * n_params)
        , n_params_(n_params)
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

        // get number of rejected models per gridpoint
        state.get_rej_len(rej_len_);

        uint_t n_gridpts = type_I_sum_.cols();
        const auto slice_size = type_I_sum_.size();
        uint_t n_params = this->n_params();

        // update type_I_sum
        for (uint_t j = 0; j < n_gridpts; ++j) {
            if (likely(rej_len_[j] == 0)) continue;

            type_I_sum_.col(j).tail(rej_len_[j]).array() += 1;
        
            // add (T - nabla_eta A(m)) for each threshold where we have rejection.
            // where T is the sufficient statistic for arm k under mean m,
            // nabla_eta A(m) is the gradient under the natural parameter eta of the log-partition function for arm k evaluated at mean m.
            auto slice_offset = 0;
            for (uint_t k = 0; k < n_params; ++k, slice_offset += slice_size) {
                Eigen::Map<mat_type<value_t> > grad_k_cache(
                        grad_sum_.data() + slice_offset, 
                        type_I_sum_.rows(),
                        type_I_sum_.cols());
                auto grad_k_j = grad_k_cache.col(j);
                grad_k_j.tail(rej_len_[j]).array() += state.get_grad(j, k);
            }
        }
    }

    /*
     * Pools quantities from another InterSum object, other,
     * as if the current object were additionally updated in the same way as in other.
     *
     * @param   other       another InterSum to pool into current object.
     */
    void pool(const InterSum& other) 
    {
        type_I_sum_ += other.type_I_sum_;
        grad_sum_ += other.grad_sum_;
        n_ += other.n_;
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
        n_params_ = n_params;
        rej_len_.setZero(n_gridpts);
        n_ = n_acc;
    }

    mat_type<uint_t>& type_I_sum() { return type_I_sum_; }
    const mat_type<uint_t>& type_I_sum() const { return type_I_sum_; }
    colvec_type<value_t>& grad_sum() { return grad_sum_; }
    const colvec_type<value_t>& grad_sum() const { return grad_sum_; }
    size_t& n_accum() { return n_; }
    size_t n_accum() const { return n_; }

    constexpr size_t n_models() const { return type_I_sum_.rows(); }
    constexpr size_t n_gridpts() const { return type_I_sum_.cols(); }
    constexpr size_t n_params() const { return n_params_; }     

private:

    mat_type<uint_t> type_I_sum_;       // Type I error sums.
                                        // type_I_sum_(i,j) = rejection accumulation for model i at gridpt j.
    colvec_type<value_t> grad_sum_;     // gradient sums.
                                        // grad_sum_(i,j,k) = partial deriv accumulation w.r.t. param k for model i at gridpt j.
    size_t n_=0;                        // number of accumulations
    size_t n_params_;

    /* Buffer needed in update for one-time allocation */
    colvec_type<uint_t> rej_len_;       // number of models that rejects for each gridpt
};

} // namespace kevlar
