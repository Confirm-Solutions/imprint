#include <iostream>
#include <kevlar_bits/grid/grid_range.hpp>
#include <kevlar_bits/grid/gridder.hpp>
#include <kevlar_bits/grid/hyperplane.hpp>
#include <kevlar_bits/grid/tile.hpp>
#include <kevlar_bits/model/normal/simple.hpp>

int main() {
    using namespace kevlar;
    using model_t = model::normal::Simple<double>;
    using tile_t = Tile<double>;
    using gr_t = GridRange<double, uint32_t, tile_t>;

    size_t n_gridpts = 100;
    size_t n_sims = 1e5;
    size_t seed = 0;
    double lower = -3.;
    double upper = 1.4;
    double alpha = 0.025;

    colvec_type<double> cvs(1);
    cvs << (upper + qnorm(1 - alpha));

    // empty null hypos
    std::vector<HyperPlane<double>> null_hypos;

    gr_t gr(1, n_gridpts);
    gr.thetas().row(0) = Gridder::make_grid(n_gridpts, lower, upper);
    double radius = Gridder::radius(n_gridpts, lower, upper);
    gr.radii().row(0).array() = radius;
    gr.sim_sizes().array() = n_sims;

    // TODO: create a default one
    gr.create_tiles(null_hypos);
    gr.prune();

    model_t model(cvs);
    auto sgs = model.make_sim_global_state<std::mt19937, double, uint32_t>(gr);
    auto ss = sgs.make_sim_state();

    colvec_type<uint32_t> rejection_length(gr.n_tiles());
    colvec_type<uint32_t> rejection_sum(gr.n_tiles());
    rejection_sum.setZero();
    std::mt19937 gen(seed);

    for (size_t i = 0; i < n_sims; ++i) {
        ss->simulate(gen, rejection_length);
        rejection_sum += rejection_length;
    }
    colvec_type<double> type_I_err =
        rejection_sum.template cast<double>() / n_sims;
    std::cout << type_I_err.transpose() << std::endl;

    return 0;
}
