#pragma once
#include <imprint_bits/util/d_ary_int.hpp>
#include <imprint_bits/util/macros.hpp>
#include <imprint_bits/util/math.hpp>
#include <imprint_bits/util/types.hpp>
#include <vector>

namespace imprint {
namespace grid {

struct AdaGridInternal {
   private:
    /*
     * Flags to indicate which action to take.
     *  - finalize_ = finalize
     *  - N_        = change sim size
     *  - eps_      = change radius
     */
    enum class action_type : char { finalize_, N_, eps_ };

    /*
     * Computes new simulation size based on old value N,
     * the factor to scale by, N_factor, and the max value, N_max.
     */
    IMPRINT_STRONG_INLINE
    constexpr static auto compute_new_sim_size(size_t N, size_t N_factor,
                                               size_t N_max) {
        return std::min(N * N_factor, N_max);
    }

   public:
    /*
     * Replaces grid_range with the new iteration of grid-points, and
     * populates grid_final with current iteration of finalized points,
     * based on the imprint bound ub.
     */
    template <class ImprintBoundType, class GridRangeType, class ValueType>
    void update(const ImprintBoundType& ub, GridRangeType& grid_range,
                GridRangeType& grid_final, size_t N_max,
                ValueType finalize_thr) const {
        using value_t = ValueType;
        using gr_t = std::decay_t<GridRangeType>;

        // allocate aux data one-time.
        const auto d = grid_range.n_params();  // dimension of grid point
        dAryInt bits(2, d);
        colvec_type<value_t> new_rad;
        colvec_type<value_t> new_pt;
        std::vector<action_type> actions(grid_range.n_tiles());

        // aliases and configuration
        const auto& ub_tot = ub.get();
        const auto N_factor = bits.n_unique();  // amount to increase sim size
        const auto n_new_pts =
            bits.n_unique();      // number of new points if eps changes
        size_t n_finalized = 0;   // number of new finalized points
        size_t n_grid_range = 0;  // number of new grid range points

        // aliases
        const auto& d0 = ub.delta_0();
        const auto& d0_u = ub.delta_0_u();
        const auto& d1 = ub.delta_1();
        const auto& d1_u = ub.delta_1_u();
        const auto& d2_u = ub.delta_2_u();
        const auto& N = grid_range.sim_sizes();

        // Note: ImprintBound rows small->large = model most->least
        // conservative. So, first row is for thr_minus and second is thr.

        // First pass through all grid points is just to determine
        // how many finalized/new grid range points we have.
        size_t pos = 0;
        for (size_t j = 0; j < grid_range.n_gridpts(); ++j) {
            // Compute Gaussian mean approximation of upper bound
            // if N changed to N*2^d, d = dimension of gridpt
            auto ss = N[j];
            auto N_new = compute_new_sim_size(ss, N_factor, N_max);
            auto N_ratio = static_cast<value_t>(ss) / N_new;

            bool any_eps = false;  // true if any tile require splitting
            bool any_N =
                false;  // true if any tile require increase in sim_size
            bool all_finalize = true;  // true if all tiles finalized

            for (size_t t = 0; t < grid_range.n_tiles(j); ++t, ++pos) {
                // Already a good estimate for ub: finalize_
                if ((ub_tot(1, pos) < finalize_thr) || (ss >= N_max)) continue;

                all_finalize = false;

                auto mu_dN =
                    d0(1, pos) + d1(1, pos) +
                    (d0_u(1, pos) + d1_u(1, pos)) * std::sqrt(N_ratio) +
                    d2_u(1, pos);

                // Compute Gaussian mean approximation of upper bound
                // if eps changed to eps/2
                auto mu_deps = d0(1, pos) + d0_u(1, pos) +
                               (d1(1, pos) + d1_u(1, pos)) / 2. +
                               d2_u(1, pos) / 4.;

                // Compare Gaussian mean approximations:
                // smaller the mean, the more likely ImprintBound < alpha.
                bool do_N = (mu_dN < mu_deps);
                any_N = any_N || do_N;
                any_eps = any_eps || !do_N;

            }  // end for-loop over tiles

            if (all_finalize) {
                actions[j] = action_type::finalize_;
                ++n_finalized;
                continue;
            }

            // prioritize splitting!
            if (any_eps) {
                actions[j] = action_type::eps_;
                n_grid_range += n_new_pts;
                continue;
            }

            // finally, if not finalize and no eps, then increase sim_size
            actions[j] = action_type::N_;
            n_grid_range += 1;

        }  // end for-loop over gridpts

        // move the current grid ranges and setup for next range.
        const gr_t grid_range_old = std::move(grid_range);
        grid_range = gr_t(d, n_grid_range);
        grid_final = gr_t(d, n_finalized);

        const auto& theta_old = grid_range_old.thetas();
        const auto& radii_old = grid_range_old.radii();
        const auto& sim_sizes_old = grid_range_old.sim_sizes();
        auto& theta_new = grid_range.thetas();
        auto& radii_new = grid_range.radii();
        auto& sim_sizes_new = grid_range.sim_sizes();
        auto& theta_fin = grid_final.thetas();
        auto& radii_fin = grid_final.radii();
        auto& sim_sizes_fin = grid_final.sim_sizes();

        // Second pass through the grid points will actually
        // populate the new grid_range and grid_final.
        size_t new_j = 0;
        size_t fin_j = 0;
        for (size_t j = 0; j < grid_range_old.n_gridpts(); ++j) {
            auto theta_old_j = theta_old.col(j);
            auto radius_old_j = radii_old.col(j);
            auto sim_size_old_j = sim_sizes_old[j];

            // finalize the point
            switch (actions[j]) {
                case action_type::finalize_: {
                    theta_fin.col(fin_j) = theta_old_j;
                    radii_fin.col(fin_j) = radius_old_j;
                    sim_sizes_fin[fin_j] = sim_size_old_j;
                    ++fin_j;
                    break;
                }

                case action_type::N_: {
                    theta_new.col(new_j) = theta_old_j;
                    radii_new.col(new_j) = radius_old_j;
                    sim_sizes_new[new_j] =
                        compute_new_sim_size(sim_size_old_j, N_factor, N_max);
                    ++new_j;
                    break;
                }

                case action_type::eps_: {
                    bits.setZero();
                    new_rad = radius_old_j / 2.;
                    for (int k = 0; k < n_new_pts; ++k, ++bits) {
                        new_pt.array() =
                            theta_old_j.array() +
                            new_rad.array() *
                                (2 * bits().cast<value_t>().array() - 1);

                        theta_new.col(new_j) = new_pt;
                        radii_new.col(new_j) = new_rad;
                        sim_sizes_new[new_j] = sim_size_old_j;
                        ++new_j;
                    }
                    break;
                }
            }  // end switch
        }
    }
};

}  // namespace grid
}  // namespace imprint
