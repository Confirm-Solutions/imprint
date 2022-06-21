#include <imprint_bits/grid/grid_range.hpp>
#include <imprint_bits/grid/gridder.hpp>
#include <imprint_bits/grid/hyperplane.hpp>
#include <imprint_bits/grid/tile.hpp>
#include <imprint_bits/model/normal/simple.hpp>
#include <iostream>

int main() {
    using namespace imprint;
    using model_t = model::normal::Simple<double>;
    using tile_t = grid::Tile<double>;
    using gr_t = grid::GridRange<double, uint32_t, tile_t>;
    using hp_t = grid::HyperPlane<double>;

    // configuration setting
    size_t n_gridpts = 100;
    size_t n_sims = 1e5;
    size_t seed = 0;
    double lower = -3.;
    double upper = 1.4;
    double alpha = 0.025;

    // initialize critical threshold
    colvec_type<double> cvs(1);
    cvs << (upper + qnorm(1 - alpha));

    // empty null hypothesis surfaces
    // we will treat all grid-points as part of the null-space.
    std::vector<hp_t> null_hypos;

    // initialize a grid range
    gr_t gr(1, n_gridpts);
    gr.thetas().row(0) = grid::Gridder::make_grid(n_gridpts, lower, upper);
    auto radius = grid::Gridder::radius(n_gridpts, lower, upper);
    gr.radii().row(0).array() = radius;
    gr.sim_sizes().array() = n_sims;

    // create tiles and prune (shouldn't affect any internals)
    gr.create_tiles(null_hypos);
    gr.prune();

    std::cout << "n_tiles: " << gr.n_tiles() << std::endl;

    // create a model object
    model_t model(cvs);

    // create a simulation global state,
    // which caches any values to speed-up simulations.
    auto sgs = model.make_sim_global_state<std::mt19937, double, uint32_t>(gr);

    // create a simulation state,
    // which defines the simulation routine.
    auto ss = sgs.make_sim_state(seed);

    colvec_type<uint32_t> rejection_length(gr.n_tiles());
    colvec_type<uint32_t> rejection_sum(gr.n_tiles());
    rejection_sum.setZero();

    // simulate and accumulate rejection counts
    for (size_t i = 0; i < n_sims; ++i) {
        ss->simulate(rejection_length);
        rejection_sum += rejection_length;
    }

    // print the Type I Error estimate
    colvec_type<double> type_I_err =
        rejection_sum.template cast<double>() / n_sims;
    std::cout << type_I_err.transpose() << std::endl;

    return 0;
}
