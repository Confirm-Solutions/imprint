#pragma once
#include <imprint_bits/util/macros.hpp>
#include <imprint_bits/util/types.hpp>

namespace imprint {
namespace bound {

/*
 * Accumulator for Type I error imprint bound.
 */
template <class ValueType, class UIntType>
struct TypeIErrorAccum {
    using value_t = ValueType;
    using uint_t = UIntType;

   private:
    mat_type<uint_t> typeI_sum_;  // Type I error sums.
                                  // typeI_sum_(i,j) = rejection accumulation
                                  // for model i at tile j.
    colvec_type<value_t>
        score_sum_;    // score sums.
                       // score_sum_(i,j,k) = partial deriv accumulation w.r.t.
                       // param j for model i at tile k.
    size_t n_params_;  // dimension of a gridpoint.

    /* Buffer needed in update for one-time allocation */
    colvec_type<value_t> score_buff_;  // score vector buffer

    IMPRINT_STRONG_INLINE void update_internal(uint_t pos, uint_t rej_len_pos) {
        typeI_sum_.col(pos).tail(rej_len_pos).array() += 1;

        const auto slice_size = n_models() * n_params();
        Eigen::Map<mat_type<value_t> > score_pos(
            score_sum_.data() + pos * slice_size, n_models(), n_params());
        for (uint_t k = 0; k < n_params_; ++k) {
            auto score_pos_k = score_pos.col(k);
            score_pos_k.tail(rej_len_pos).array() += score_buff_(k);
        }
    }

   public:
    TypeIErrorAccum() = default;
    TypeIErrorAccum(size_t n_models, size_t n_tiles, size_t n_params)
        : typeI_sum_(n_models, n_tiles),
          score_sum_(n_models * n_params * n_tiles),
          n_params_(n_params),
          score_buff_(n_params) {
        typeI_sum_.setZero();
        score_sum_.setZero();
    }

    /*
     * Accumulates estimates based on current model SimState.
     * Increments typeI_sum by rejection indicators.
     * Increments score_sum by rejection indicators * (T - \nabla A).
     *
     * @param   rej_len             rej_len[i] = number of models that rejected
     * at tile i.
     * @param   sim_state           SimState-like object that was used to
     * produce rej_len. Assumes that the sequence of models considered is
     * ordered in ascending order, in the sense that if ith model rejects then
     * jth model rejects for all j >= i.
     * @param   grid_range          a grid range object on which sim_state ran
     * its simulation to produce rej_len.
     */
    template <class VecType, class SimStateType, class GridRangeType>
    void update(const VecType& rej_len, const SimStateType& sim_state,
                const GridRangeType& grid_range) {
        assert(grid_range.n_tiles() == typeI_sum_.cols());
        assert(grid_range.n_params() == n_params_);
        assert(score_buff_.size() == n_params_);

        const auto& gr_view = grid_range;
        const uint_t n_gridpts = gr_view.n_gridpts();

        // update typeI_sum_ and score_sum_
        size_t pos = 0;
        for (uint_t i = 0; i < n_gridpts; ++i) {
            // if current gridpoint is regular,
            // only update if there is any rejection
            if (gr_view.is_regular(i)) {
                if (unlikely(rej_len[pos] != 0)) {
                    sim_state.score(i, score_buff_);
                    update_internal(pos, rej_len[pos]);
                }
                ++pos;
                continue;
            }

            // then iterate through all the tiles for update
            bool score_computed = false;
            const auto n_ts = gr_view.n_tiles(i);
            for (uint_t j = 0; j < n_ts; ++j, ++pos) {
                if (unlikely(rej_len[pos] == 0)) continue;
                if (!score_computed) {
                    score_computed = true;
                    sim_state.score(i, score_buff_);
                }
                update_internal(pos, rej_len[pos]);
            }
        }
    }

    /*
     * Pools quantities from another TypeIErrorAccum object, other,
     * as if the current object were additionally updated in the same way as in
     * other.
     *
     * @param   other       another TypeIErrorAccum to pool into current object.
     */
    void pool(const TypeIErrorAccum& other) {
        typeI_sum_ += other.typeI_sum_;
        score_sum_ += other.score_sum_;
    }

    /*
     * Pools with the raw sum and score from another accumulation. as if the
     * current object were additionally updated in the same way as in other.
     *
     * @param   other_typeI_sum       the other type I sum
     * @param   other_typeI_score       the other type I score
     */
    void pool_raw(const mat_type<uint_t>& other_typeI_sum,
                  const colvec_type<value_t>& other_typeI_score) {
        typeI_sum_ += other_typeI_sum;
        score_sum_ += other_typeI_score;
    }

    /*
     * Reset the size of internal data structures corresponding
     * to the new configuration n_models, n_tiles, n_params, n_acc.
     * The first three parameters must be positive.
     *
     * @param   n_models        number of models.
     * @param   n_tiles         number of tiles.
     * @param   n_params        number of parameters.
     */
    void reset(size_t n_models, size_t n_tiles, size_t n_params) {
        typeI_sum_.setZero(n_models, n_tiles);
        score_sum_.setZero(n_models * n_params * n_tiles);
        score_buff_.resize(n_params);
        n_params_ = n_params;
    }

    const mat_type<uint_t>& typeI_sum() const { return typeI_sum_; }
    const colvec_type<value_t>& score_sum() const { return score_sum_; }

    constexpr size_t n_tiles() const { return typeI_sum_.cols(); }
    constexpr size_t n_params() const { return n_params_; }
    constexpr size_t n_models() const { return typeI_sum_.rows(); }

    // helper debug functions that should not be used by average users.
    mat_type<uint_t>& typeI_sum__() { return typeI_sum_; }
    colvec_type<value_t>& score_sum__() { return score_sum_; }
};

}  // namespace bound
}  // namespace imprint
