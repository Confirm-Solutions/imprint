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
        size_t n_tiles,
        size_t n_params)
        : type_I_sum_(n_models, n_tiles)
        , grad_sum_(n_models * n_tiles * n_params)
        , n_params_(n_params)
        , rej_len_(n_tiles)
        , grad_buff_(n_params)
    {
        type_I_sum_.setZero();
        grad_sum_.setZero();
        rej_len_.setZero();
        grad_buff_.setZero();
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
        // get number of rejected models per gridpoint
        state.rej_len(rej_len_);

        const auto& gr_view = state.grid_range();
        const uint_t n_gridpts = gr_view.n_gridpts();

        // update type_I_sum_ and grad_sum_
        size_t pos = 0;
        for (uint_t i = 0; i < n_gridpts; ++i) {

            // if current gridpoint is regular,
            // only update if there is any rejection
            if (gr_view.is_regular(i)) {
                if (unlikely(rej_len_[pos] != 0)) {
                    update_internal(pos, [&](uint_t k) { return state.grad(i,k); });
                }
                ++pos;
                continue;
            }

            // then iterate through all the tiles for update
            bool grad_computed = false;
            const auto n_ts = gr_view.n_tiles(i);
            for (uint_t j = 0; j < n_ts; ++j, ++pos) {
                if (unlikely(rej_len_[pos] == 0)) continue;
                if (!grad_computed) {
                    grad_computed = true;
                    for (uint_t k = 0; k < grad_buff_.size(); ++k) {
                        grad_buff_[k] = state.grad(i, k);
                    }
                }
                update_internal(pos, [&](uint_t k) { return grad_buff_[k]; });
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
    }

    /*
     * Reset the size of internal data structures corresponding
     * to the new configuration n_models, n_tiles, n_params, n_acc.
     * The first three parameters must be positive.
     *
     * @param   n_models        number of models.
     * @param   n_tiles       number of tiles.
     * @param   n_params        number of parameters.
     */
    void reset(
        size_t n_models,
        size_t n_tiles,
        size_t n_params)
    {
        type_I_sum_.setZero(n_models, n_tiles);
        grad_sum_.setZero(n_models * n_tiles * n_params);
        n_params_ = n_params;
        rej_len_.setZero(n_tiles);
    }

    mat_type<uint_t>& type_I_sum() { return type_I_sum_; }
    const mat_type<uint_t>& type_I_sum() const { return type_I_sum_; }
    colvec_type<value_t>& grad_sum() { return grad_sum_; }
    const colvec_type<value_t>& grad_sum() const { return grad_sum_; }

    constexpr size_t n_tiles() const { return type_I_sum_.cols(); }
    constexpr size_t n_params() const { return n_params_; }     

private:

    template <class F>
    KEVLAR_STRONG_INLINE
    void update_internal(
            uint_t pos, 
            F get_grad)
    {
        type_I_sum_.col(pos).tail(rej_len_[pos]).array() += 1;
    
        // add (T - nabla_eta A(m)) for each threshold where we have rejection.
        // where T is the sufficient statistic for arm k under mean m,
        // nabla_eta A(m) is the gradient under the natural parameter eta of the log-partition function for arm k evaluated at mean m.
        const auto slice_size = type_I_sum_.size();
        uint_t slice_offset = 0;
        for (uint_t k = 0; k < n_params_; ++k, slice_offset += slice_size) {
            Eigen::Map<mat_type<value_t> > grad_k_cache(
                    grad_sum_.data() + slice_offset, 
                    type_I_sum_.rows(),
                    type_I_sum_.cols());
            auto grad_k_j = grad_k_cache.col(pos);
            grad_k_j.tail(rej_len_[pos]).array() += get_grad(k);
        }
    }

    mat_type<uint_t> type_I_sum_;       // Type I error sums.
                                        // type_I_sum_(i,j) = rejection accumulation for model i at tile j.
    colvec_type<value_t> grad_sum_;     // gradient sums.
                                        // grad_sum_(i,j,k) = partial deriv accumulation w.r.t. param k for model i at tile j.
    size_t n_params_;                   // dimension of a gridpoint.

    /* Buffer needed in update for one-time allocation */
    colvec_type<uint_t> rej_len_;       // number of models that rejects for each gridpt
    colvec_type<value_t> grad_buff_;    // gradient vector buffer for general tile case
};

} // namespace kevlar
