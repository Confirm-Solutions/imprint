#pragma once
#include <Eigen/Core>
#include <kevlar_bits/util/math.hpp>
#include <kevlar_bits/util/types.hpp>
#include <kevlar_bits/util/macros.hpp>
#include <stats.hpp>

namespace kevlar {

/*
 * This class encapsulates the logic of constructing an upper bound.
 * It stores all necessary components of the upper bound.
 *
 * @param   ValueType       underlying value type (usually double).
 */
template <class ValueType>
struct UpperBound {
    using value_t = ValueType;

    /*
     * Creates and stores the components to the upper bound estimate.
     *
     * @param   model               ModelBase-like object.
     * @param   is_o                InterSum-like object.
     *                              Must be an object updated from simulating
     * under model attached with grid_range.
     * @param   grid_range          GridRange-like object.
     *                              Must be the same grid range that is_o
     *                              was updated with.
     * @param   delta               confidence of provable upper bound.
     * @param   delta_prop_0to1     proportion of delta to put
     *                              into 0th order upper bound.
     *                              Default is 0.5.
     * @param   verbose             if true, then more quantities will be saved.
     *                              Currently, it stores the corner points of
     * each tile that maximizes the d1 + d1u + d2u. Note that for regular
     * gridpoints, the saved vertices are undefined. Default is false.
     */
    template <class ModelType, class InterSumType, class GridRangeType>
    void create(const ModelType& model, const InterSumType& is_o,
                const GridRangeType& grid_range, value_t delta,
                value_t delta_prop_0to1 = 0.5, bool verbose = false) {
        const auto n_params = grid_range.n_params();
        const auto n_tiles = is_o.n_tiles();
        const auto n_models = model.n_models();

        if (verbose) {
            const auto slice_size = n_params * n_tiles;
            vertices_.resize(slice_size * n_models);
            create_internal(model, is_o, grid_range, delta, delta_prop_0to1,
                            [&](size_t m, size_t pos, const auto& v) {
                                Eigen::Map<mat_type<value_t> > cor_slice(
                                    vertices_.data() + slice_size * m, n_params,
                                    n_tiles);
                                cor_slice.col(pos) = v;
                            });
        } else {
            create_internal(model, is_o, grid_range, delta, delta_prop_0to1,
                            [](size_t, size_t, const auto&) {});
        }

        full_ = delta_0_ + delta_0_u_ + delta_1_ + delta_1_u_ + delta_2_u_;
    }

    /*
     * Returns the total upper bound computed from the components.
     */
    const mat_type<value_t>& get() const { return full_; }

    /*
     * Returns the vertices that maximize d1 + d1u + d2u
     * for each model and tile.
     */
    colvec_type<value_t> vertices() const { return vertices_; }

    mat_type<value_t>& delta_0() { return delta_0_; }
    mat_type<value_t>& delta_0_u() { return delta_0_u_; }
    mat_type<value_t>& delta_1() { return delta_1_; }
    mat_type<value_t>& delta_1_u() { return delta_1_u_; }
    mat_type<value_t>& delta_2_u() { return delta_2_u_; }
    const mat_type<value_t>& delta_0() const { return delta_0_; }
    const mat_type<value_t>& delta_0_u() const { return delta_0_u_; }
    const mat_type<value_t>& delta_1() const { return delta_1_; }
    const mat_type<value_t>& delta_1_u() const { return delta_1_u_; }
    const mat_type<value_t>& delta_2_u() const { return delta_2_u_; }

   private:
    template <class ModelType, class InterSumType, class GridRangeType,
              class SaveCornerType>
    void create_internal(const ModelType& model, const InterSumType& is_o,
                         const GridRangeType& grid_range, value_t delta,
                         value_t delta_prop_0to1, SaveCornerType save_corner) {
        // some aliases
        const auto n_models = model.n_models();
        const auto n_gridpts = grid_range.n_gridpts();
        const auto n_tiles = is_o.n_tiles();  // total number of tiles
        const auto n_params = grid_range.n_params();
        const auto slice_size = n_models * n_tiles;
        const auto& sim_sizes = grid_range.sim_sizes();
        const auto& typeIsum = is_o.type_I_sum();
        const auto& thetas = grid_range.thetas();
        const auto& tiles = grid_range.tiles();
        constexpr value_t neg_inf = -std::numeric_limits<value_t>::infinity();

        // pre-compute some constants
        const value_t d0u_factor = 1. - delta * delta_prop_0to1;
        const value_t d1u_factor =
            std::sqrt(1. / ((1.0 - delta_prop_0to1) * delta) - 1.);

        // populate 0th order and upper bound
        delta_0_.resize(n_models, n_tiles);
        delta_0_u_.resize(n_models, n_tiles);
        delta_1_.setZero(n_models, n_tiles);
        delta_1_u_.resize(n_models, n_tiles);
        delta_2_u_.resize(n_models, n_tiles);

        colvec_type<value_t> d11u2u(n_models);  // d1 + d1u + d2u for each model
        colvec_type<value_t> v_diff(n_params);  // buffer to store vertex-gridpt
        colvec_type<value_t> deta_v_diff;       // Deta^T v_diff

        size_t pos = 0;
        for (size_t gp = 0; gp < n_gridpts; ++gp) {
            const auto ss = sim_sizes[gp];
            const auto sqrt_ss = std::sqrt(ss);
            const auto d1u_factor_sqrt_ss = d1u_factor / sqrt_ss;

            for (size_t i = 0; i < grid_range.n_tiles(gp); ++i, ++pos) {
                // update 0th order
                auto delta_0_j = delta_0_.col(pos);
                auto typeIsum_j = typeIsum.col(pos);
                delta_0_j = typeIsum_j.template cast<value_t>() / ss;

                // update 0th order upper
                auto delta_0_u_j = delta_0_u_.col(pos);
                for (size_t m = 0; m < delta_0_u_j.size(); ++m) {
                    delta_0_u_j[m] =
                        ::stats::qbeta(d0u_factor, typeIsum_j[m] + 1,
                                       ss - typeIsum_j[m]) -
                        delta_0_j[m];
                }

                // update 1st/1st upper/2nd upper
                const auto& tile = tiles[pos];

                // set current max value of d1 + d1u + d2u = -inf for all
                // models.
                d11u2u.fill(neg_inf);

                // iterate over all vertices of the tile
                // and update current max of d1 + d1u + d2u
                // and d1, d1u, d2u that achieve that max.
                if (grid_range.is_regular(gp)) {
                    update_d11u2u(tile.begin_full(), tile.end_full(), true, gp,
                                  pos, ss, d1u_factor_sqrt_ss, v_diff,
                                  deta_v_diff, thetas, model, n_models, n_tiles,
                                  slice_size, is_o, d11u2u, save_corner);
                } else {
                    update_d11u2u(tile.begin(), tile.end(), false, gp, pos, ss,
                                  d1u_factor_sqrt_ss, v_diff, deta_v_diff,
                                  thetas, model, n_models, n_tiles, slice_size,
                                  is_o, d11u2u, save_corner);
                }
            }  // end for-loop on tiles
        }
    }

    template <class Iter, class VDiffType, class DetaVDiffType,
              class ThetasType, class ModelType, class ISType, class D11U2UType,
              class SaveCornerType>
    void update_d11u2u(Iter begin, Iter end, bool is_reg, size_t gp, size_t pos,
                       size_t ss, value_t d1u_factor_sqrt_ss, VDiffType& v_diff,
                       DetaVDiffType& deta_v_diff, const ThetasType& thetas,
                       const ModelType& model, size_t n_models, size_t n_tiles,
                       size_t slice_size, const ISType& is_o,
                       D11U2UType& d11u2u, SaveCornerType save_corner) {
        for (; begin != end; ++begin) {
            auto&& v = *begin;  // vertex

            v_diff = v - thetas.col(gp);
            model.eta_transform(gp, v_diff, deta_v_diff);
            value_t d1u =
                std::sqrt(model.cov_quad(gp, deta_v_diff)) * d1u_factor_sqrt_ss;
            value_t d2u = model.max_cov_quad(gp, v_diff);
            d2u += v_diff.squaredNorm() * model.max_eta_hess_cov(gp);
            d2u *= 0.5;

            for (size_t m = 0; m < n_models; ++m) {
                // compute current v^T grad_f
                value_t d1 = 0;
                size_t slice_offset = 0;
                for (size_t k = 0; k < deta_v_diff.size();
                     ++k, slice_offset += slice_size) {
                    Eigen::Map<const mat_type<value_t> > grad_k(
                        is_o.grad_sum().data() + slice_offset, n_models,
                        n_tiles);
                    d1 += deta_v_diff[k] * grad_k(m, pos);
                }
                d1 /= ss;

                // check if we have new maximum
                value_t new_max = d1 + d1u + d2u;
                bool is_new = (new_max > d11u2u[m]);

                // save new maximum sum and the components
                if (is_new) {
                    d11u2u[m] = new_max;
                    delta_1_(m, pos) = d1;
                    delta_1_u_(m, pos) = d1u;
                    delta_2_u_(m, pos) = d2u;
                    save_corner(m, pos, v);
                }

            }  // end for-loop on models
        }      // end for-loop on vertices
    }

    // Components that make up an upper bound.
    mat_type<value_t> delta_0_;  // 0th order (n_models x n_gridpts)
    mat_type<value_t>
        delta_0_u_;              // 0th order upper bound (n_models x n_gridpts)
    mat_type<value_t> delta_1_;  // 1st order (n_models x n_gridpts)
    mat_type<value_t>
        delta_1_u_;  // 1st order upper bound (n_models x n_gridpts)
    mat_type<value_t>
        delta_2_u_;           // 2nd order upper bound (n_models x n_gridpts)
    mat_type<value_t> full_;  // full upper bound = sum of previous components

    colvec_type<value_t>
        vertices_;  // vertices that achieve the maximum of d1+d1u+d2u
                    // n_params x n_gridpts x n_models
                    // Note that the structure is slightly different
                    // from InterSum gradient sum (3D) array
                    // (n_models x n_gridpts x n_params).
                    // This ordering allows us to return a viewer of
                    // each corner without making a copy,
                    // and saving each corner can be vectorized.
};

}  // namespace kevlar
