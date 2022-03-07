#pragma once
#include <kevlar_bits/util/types.hpp>
#include <kevlar_bits/util/math.hpp>
#include <kevlar_bits/util/d_ary_int.hpp>
#include <vector>

namespace kevlar {

template <class ValueType>
struct AdaGridInternal
{
private:
    /*
     * Flags to indicate which action to take.
     *  - finalize_ = finalize
     *  - N_        = change sim size
     *  - eps_      = change radius
     */
    enum class action_type : char
    {
        finalize_,
        N_,
        eps_
    };

public:
    using value_t = ValueType;

    /*
     * Replaces grid_range with the new iteration of grid-points, and
     * populates grid_final with current iteration of finalized points,
     * based on the upper bound ub.
     */
    template <class UpperBoundType
            , class GridRangeType
            , class GridFinalType
            , class IsNotAltType>
    void update(const UpperBoundType& ub,
                GridRangeType&& grid_range,
                GridFinalType&& grid_final,
                IsNotAltType is_not_alt,
                uint32_t N_max,
                value_t finalize_thr) const
    {
        using gr_t = std::decay_t<GridRangeType>;
        using gf_t = std::decay_t<GridFinalType>;

        std::vector<action_type> actions(grid_range.size());
        mat_type<value_t> ub_tot = ub.get();

        const uint32_t d = grid_range.n_params();  // dimension of grid point
        const uint32_t N_factor = ipow(2, d); // amount to increase sim size
        const uint32_t n_new_pts = ipow(2, d);    // number of new points if eps changes
        uint32_t n_finalized = 0;   // number of new finalized points
        uint32_t n_grid_range = 0;  // number of new grid range points

        // aliases
        auto& d0 = ub.delta_0();
        auto& d0_u = ub.delta_0_u();
        auto& d1 = ub.delta_1();
        auto& d1_u = ub.delta_1_u();
        auto& d2_u = ub.delta_2_u();
        auto& N = grid_range.sim_sizes(); 
        auto& radii = grid_range.radii();
        auto& thetas = grid_range.thetas();

        // allocate aux data one-time.
        dAryInt bits(2, d);
        colvec_type<value_t> new_rad;
        colvec_type<value_t> new_pt;

        // Note: UpperBound rows small->large = model most->least conservative.
        // So, first row is for thr_minus and second is thr.
        
        // First pass through all grid points is just to determine
        // how many finalized/new grid range points we have.
        for (size_t j = 0; j < grid_range.size(); ++j) 
        {
            // Already a good estimate for ub: finalize_
            if ((ub_tot(1,j) < finalize_thr) || (N[j] >= N_max)) {
                actions[j] = action_type::finalize_;
                ++n_finalized;
                continue;
            }

            // Compute Gaussian mean approximation of upper bound
            // if N changed to N*2^d, d = dimension of gridpt
            auto N_new = compute_new_sim_size(N[j], N_factor, N_max);
            auto N_ratio = static_cast<value_t>(N[j])/N_new;

            auto mu_dN = d0(1,j) + d1(1,j) 
                + (d0_u(1,j) + d1_u(1,j)) * std::sqrt(N_ratio) 
                + d2_u(1,j);

            // Compute Gaussian mean approximation of upper bound
            // if eps changed to eps/2
            auto mu_deps = d0(1,j) 
                + d0_u(1,j) 
                + (d1(1,j) + d1_u(1,j)) / 2. 
                + d2_u(1,j) / 4.;

            // Compare Gaussian mean approximations:
            // smaller the mean, the more likely UpperBound < alpha.
            bool do_N = (mu_dN < mu_deps);
            actions[j] = do_N ?
                action_type::N_ : action_type::eps_;

            n_grid_range += do_N;

            if (!do_N) {
                bits.setZero();
                new_rad = radii.col(j) / 2.;

                // TODO: this is where we'll be smarter about picking the
                // vertices of the hyperplane intersected with null hypothesis.
                for (size_t k = 0; k < n_new_pts; ++k, ++bits) 
                {
                    new_pt.array() = thetas.col(j).array() + 
                        new_rad.array() *
                            (2*bits().cast<value_t>().array()-1);

                    // only add the new point if it's "viable".
                    // The only check right now is 
                    // if it's not in alternative space.
                    n_grid_range += !!is_not_alt(new_pt);
                }
            }
        }

        // move the current grid ranges and setup for next range.
        const gr_t grid_range_old = std::move(grid_range);
        grid_range = gr_t(d, n_grid_range);
        grid_final = gf_t(d, n_finalized);

        auto gr_old_it = grid_range_old.begin();
        auto gr_it = grid_range.begin();
        auto gf_it = grid_final.begin();

        // Second pass through the grid points will actually
        // populate the new grid_range and grid_final.
        for (size_t j = 0; j < grid_range_old.size(); ++j, ++gr_old_it) 
        {
            auto old_view = *gr_old_it;

            // finalize the point
            switch(actions[j]) 
            {
                case action_type::finalize_: 
                {
                    auto view = *gf_it;
                    view.theta() = old_view.theta();
                    view.radius() = old_view.radius();
                    view.sim_size() = old_view.sim_size();
                    ++gf_it;
                    break;
                }

                case action_type::N_:
                {
                    auto view = *gr_it;
                    view.theta() = old_view.theta();
                    view.radius() = old_view.radius();
                    view.sim_size() = compute_new_sim_size(
                            old_view.sim_size(), N_factor, N_max);
                    ++gr_it;
                    break;
                }

                case action_type::eps_:
                {
                    bits.setZero();
                    new_rad = old_view.radius() / 2.;
                    for (size_t k = 0; k < n_new_pts; ++k, ++bits) 
                    {
                        new_pt.array() = old_view.theta().array() + 
                            new_rad.array() *
                                (2*bits().cast<value_t>().array()-1);

                        // only add the new point if it's "viable".
                        // The only check right now is 
                        // if it's not in alternative space.
                        if (is_not_alt(new_pt)) {
                            auto view = *gr_it; 
                            view.theta() = new_pt;
                            view.radius() = new_rad;
                            view.sim_size() = old_view.sim_size();
                            ++gr_it;
                        }
                    }
                    break;
                }
            } // end switch
        }
    }

private:

    /*
     * Computes new simulation size based on old value N,
     * the factor to scale by, N_factor, and the max value, N_max.
     */
    constexpr static auto compute_new_sim_size(
            size_t N,
            size_t N_factor,
            size_t N_max) 
    {
        return std::min(N * N_factor, N_max);
    }
};

} // namespace kevlar 
