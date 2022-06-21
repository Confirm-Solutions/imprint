#include <imprint_bits/bound/accumulator/typeI_error_accum.hpp>
#include <imprint_bits/driver/accumulate.hpp>
#include <imprint_bits/grid/grid_range.hpp>
#include <imprint_bits/grid/gridder.hpp>
#include <imprint_bits/grid/hyperplane.hpp>
#include <imprint_bits/grid/tile.hpp>
#include <imprint_bits/model/binomial/simple_selection.hpp>
#include <imprint_bits/util/algorithm.hpp>

#include "benchmark/benchmark.h"

namespace imprint {

struct binomial_fixture : benchmark::Fixture {
    using gen_t = std::mt19937;
    using value_t = double;
    using uint_t = uint32_t;
    using grid_t = grid::Gridder;
    using tile_t = grid::Tile<value_t>;
    using grid_range_t = grid::GridRange<value_t, uint_t, tile_t>;
    using hp_t = grid::HyperPlane<value_t>;
    using model_t = model::binomial::SimpleSelection<double>;
    using acc_t = bound::TypeIErrorAccum<value_t, uint_t>;

    size_t n_thetas_1d = 64;
    double lower = -0.5;
    double upper = 0.5;
    size_t n_sim = 1e5;
    double alpha = 0.025;
    double delta = 0.025;
    size_t grid_dim = 3;
    size_t n_samples = 250;
    size_t ph2_size = 50;
    double thresh = 1.96;
    size_t n_threads = std::thread::hardware_concurrency();
};

BENCHMARK_DEFINE_F(binomial_fixture, bench_fit)(benchmark::State& state) {
    size_t grid_radius = grid_t::radius(n_thetas_1d, lower, upper);

    colvec_type<value_t> theta_1d;
    Eigen::VectorXd thresholds;

    // initialize threshold
    thresholds.resize(1);
    thresholds << thresh;

    // define hyperplanes
    std::vector<hp_t> hps;
    for (size_t k = 1; k < grid_dim; ++k) {
        colvec_type<value_t> normal(grid_dim);
        normal.setZero();
        normal[0] = 1;
        normal[k] = -1;
        hps.emplace_back(normal, 0);
    }

    // create grid
    theta_1d =
        grid_t::make_grid(n_thetas_1d, lower, upper).template cast<value_t>();
    dAryInt bits(n_thetas_1d, grid_dim);
    grid_range_t grid_range(grid_dim, bits.n_unique());
    for (size_t j = 0; j < bits.n_unique(); ++j, ++bits) {
        for (size_t i = 0; i < grid_dim; ++i) {
            grid_range.thetas()(i, j) = theta_1d[bits()(i)];
        }
    }
    grid_range.radii().array() = grid_radius;
    grid_range.sim_sizes().array() = n_sim;

    grid_range.create_tiles(hps);
    grid_range.prune();

    model_t model(grid_dim, n_samples, ph2_size, thresholds);
    auto sgs = model.make_sim_global_state<gen_t, value_t, uint_t>(grid_range);

    acc_t acc_o(model.n_models(), grid_range.n_tiles(), grid_range.n_params());

    for (auto _ : state) {
        driver::accumulate(sgs, grid_range, acc_o, n_sim, 0, n_threads);
    }
}

BENCHMARK_REGISTER_F(binomial_fixture, bench_fit);

}  // namespace imprint
