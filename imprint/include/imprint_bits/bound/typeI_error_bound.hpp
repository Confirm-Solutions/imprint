#pragma once
#include <Eigen/Dense>
#include <imprint_bits/util/macros.hpp>
#include <imprint_bits/util/math.hpp>
#include <imprint_bits/util/types.hpp>
//#include <stats.hpp>  // third-party

namespace imprint {
namespace bound {

/*
 * This class encapsulates the logic of constructing
 * a Type I error imprint bound.
 * It stores all necessary components of the imprint bound.
 *
 * @param   ValueType       underlying value type (usually double).
 */
template <class ValueType>
struct TypeIErrorBound {
    using value_t = ValueType;

   private:
    // Components that make up the imprint bound.
    mat_type<value_t> delta_0_;    // 0th order (n_models x n_tiles)
    mat_type<value_t> delta_0_u_;  // 0th order upper bound (n_models x n_tiles)
    mat_type<value_t> delta_1_;    // 1st order (n_models x n_tiles)
    mat_type<value_t> delta_1_u_;  // 1st order upper bound (n_models x n_tiles)
    mat_type<value_t> delta_2_u_;  // 2nd order upper bound (n_models x n_tiles)
    mat_type<value_t> full_;  // full upper bound = sum of previous components

    colvec_type<value_t>
        vertices_;  // vertices that achieve the maximum of
                    // delta_1_ + delta_1_u_ + delta_2_u_.
                    // It is of shape (n_params, n_tiles, n_models).
                    // Note that the structure is slightly different
                    // from the accumulator score sum (3D) array
                    // (n_models, n_params, n_tiles).
                    // This ordering allows us to return a viewer of
                    // each corner without making a copy,
                    // and saving each corner can be vectorized.

    template <class KBSType, class AccumType, class GridRangeType,
              class SaveCornerType>
    void create_internal(KBSType&& kbs, const AccumType& acc_o,
                         const GridRangeType& grid_range, value_t delta,
                         value_t delta_prop_0to1, SaveCornerType save_corner) {
        // some aliases
        const auto n_models = acc_o.n_models();
        const auto n_gridpts = grid_range.n_gridpts();
        const auto n_tiles = acc_o.n_tiles();  // total number of tiles
        const auto n_params = grid_range.n_params();
        const auto n_nat_params =
            kbs.n_natural_params();  // number of natural params
        const auto slice_size = n_models * n_params;
        const auto& sim_sizes = grid_range.sim_sizes();
        const auto& typeIsum = acc_o.typeI_sum();
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
        colvec_type<value_t> deta_v_diff(n_nat_params);  // Deta * v_diff

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
                for (int m = 0; m < delta_0_u_j.size(); ++m) {
                    delta_0_u_j[m] = ibeta_inv(typeIsum_j[m] + 1,
                                               ss - typeIsum_j[m], d0u_factor) -
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
                    update_d11u2u(tile, tile.begin_full(), tile.end_full(),
                                  true, gp, pos, ss, d1u_factor_sqrt_ss, v_diff,
                                  deta_v_diff, thetas, kbs, n_models, n_params,
                                  slice_size, acc_o, d11u2u, save_corner);
                } else {
                    update_d11u2u(tile, tile.begin(), tile.end(), false, gp,
                                  pos, ss, d1u_factor_sqrt_ss, v_diff,
                                  deta_v_diff, thetas, kbs, n_models, n_params,
                                  slice_size, acc_o, d11u2u, save_corner);
                }
            }  // end for-loop on tiles
        }
    }

    template <class TileType, class Iter, class VDiffType, class DetaVDiffType,
              class ThetasType, class KBSType, class AccumType,
              class D11U2UType, class SaveCornerType>
    void update_d11u2u(const TileType& tile, Iter begin, Iter end, bool is_reg,
                       size_t gp, size_t tile_pos, size_t ss,
                       value_t d1u_factor_sqrt_ss, VDiffType& v_diff,
                       DetaVDiffType& deta_v_diff, const ThetasType& thetas,
                       KBSType&& kbs, size_t n_models, size_t n_params,
                       size_t slice_size, const AccumType& acc_o,
                       D11U2UType& d11u2u, SaveCornerType save_corner) {
        Eigen::Map<const mat_type<value_t> > score_tile(
            acc_o.score_sum().data() + slice_size * tile_pos, n_models,
            n_params);

        for (; begin != end; ++begin) {
            auto&& v = *begin;  // vertex

            auto center = thetas.col(gp);
            v_diff = v - center;
            kbs.apply_eta_jacobian(gp, v_diff, deta_v_diff);
            value_t d1u = std::sqrt(kbs.covar_quadform(gp, deta_v_diff)) *
                          d1u_factor_sqrt_ss;
            value_t d2u =
                0.5 * kbs.hessian_quadform_bound(gp, tile_pos, v_diff);

            for (size_t m = 0; m < n_models; ++m) {
                // compute current v^T Deta^T score
                value_t d1 = score_tile.row(m).dot(deta_v_diff) / ss;

                // check if we have new maximum
                value_t new_max = d1 + d1u + d2u;
                bool is_new = (new_max > d11u2u[m]);

                // save new maximum sum and the components
                if (is_new) {
                    d11u2u[m] = new_max;
                    delta_1_(m, tile_pos) = d1;
                    delta_1_u_(m, tile_pos) = d1u;
                    delta_2_u_(m, tile_pos) = d2u;
                    save_corner(m, tile_pos, v);
                }

            }  // end for-loop on models
        }      // end for-loop on vertices
    }

   public:
    /*
     * Creates and stores the components of the imprint bound.
     *
     * @param   kbs                 ImprintBoundState-like object.
     * @param   acc_o               Accumulator object for Type I error.
     *                              Assumes that this is an accumulation
     *                              of simulations for the SimState associated
     *                              with the same model class as the one that
     *                              generated kbs.
     * @param   grid_range          GridRange-like object.
     *                              Assumes that this is the same grid range
     *                              that acc_o was updated with and that kbs
     *                              is initialized with.
     * @param   delta               1-confidence of provable upper bound.
     * @param   delta_prop_0to1     proportion of delta to put
     *                              into 0th order upper bound.
     *                              Default is 0.5.
     * @param   verbose             if true, then more quantities will be saved.
     *                              Currently, it stores the corner points of
     * each tile that maximizes the
     *  first order + first order upper bound + second order upper bound
     * Note that for regular gridpoints,
     * the saved vertices are undefined. Default is false.
     */
    template <class KBStateType, class KBSAccType, class GridRangeType>
    void create(KBStateType&& kbs, const KBSAccType& acc_o,
                const GridRangeType& grid_range, value_t delta,
                value_t delta_prop_0to1 = 0.5, bool verbose = false) {
        const auto n_params = grid_range.n_params();
        const auto n_tiles = grid_range.n_tiles();
        const auto n_models = acc_o.n_models();

        if (verbose) {
            const auto slice_size = n_params * n_tiles;
            vertices_.resize(slice_size * n_models);
            create_internal(kbs, acc_o, grid_range, delta, delta_prop_0to1,
                            [&](size_t m, size_t pos, const auto& v) {
                                Eigen::Map<mat_type<value_t> > cor_slice(
                                    vertices_.data() + slice_size * m, n_params,
                                    n_tiles);
                                cor_slice.col(pos) = v;
                            });
        } else {
            create_internal(kbs, acc_o, grid_range, delta, delta_prop_0to1,
                            [](size_t, size_t, const auto&) {});
        }

        full_ = delta_0_ + delta_0_u_ + delta_1_ + delta_1_u_ + delta_2_u_;
    }

    /*
     * Returns the total upper bound computed from the components.
     */
    const mat_type<value_t>& get() const { return full_; }

    /*
     * Returns the vertices that maximize
     *  first order + first order upper bound + second order upper bound
     * for each model and tile.
     * The output is a vector representing a 3-D array of shape
     * (n_params, n_tiles, n_models), so that vertices()[:,j,k]
     * is the maximizing vertex at jth tile and kth model.
     */
    const colvec_type<value_t>& vertices() const { return vertices_; }

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
};

}  // namespace bound
}  // namespace imprint
