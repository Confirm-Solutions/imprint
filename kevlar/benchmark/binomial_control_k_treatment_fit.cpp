#include <kevlar_bits/driver/fit.hpp>
#include <kevlar_bits/grid/grid_range.hpp>
#include <kevlar_bits/grid/gridder.hpp>
#include <kevlar_bits/grid/hyperplane.hpp>
#include <kevlar_bits/grid/tile.hpp>
#include <kevlar_bits/model/binomial_control_k_treatment.hpp>
#include <kevlar_bits/stats/inter_sum.hpp>
#include <kevlar_bits/util/algorithm.hpp>

#include "benchmark/benchmark.h"

namespace kevlar {

struct binomial_fixture : benchmark::Fixture {
    using grid_t = Gridder;
    using tile_t = Tile<double>;
    using grid_range_t = GridRange<double, uint32_t, tile_t>;
    using hp_t = HyperPlane<double>;
    using bckt_t = BinomialControlkTreatment<double, uint32_t, grid_range_t>;
    using is_t = InterSum<double, uint32_t>;

    size_t n_thetas_1d = 32;
    double lower = -0.5;
    double upper = 1.5;
    size_t n_sim = 100;
    double alpha = 0.025;
    double delta = 0.025;
    size_t grid_dim = 2;
    size_t grid_radius = grid_t::radius(n_thetas_1d, lower, upper);
    size_t n_samples = 250;
    size_t ph2_size = 50;
    double thresh = 1.96;
    Eigen::VectorXd theta_1d;
    Eigen::VectorXd thresholds;
};

BENCHMARK_DEFINE_F(binomial_fixture, bench_fit)(benchmark::State& state) {
    // initialize threshold
    thresholds.resize(1);
    thresholds << thresh;

    // define hyperplanes
    std::vector<hp_t> hps;
    colvec_type<double> normal(grid_dim);
    normal << 1, -1;
    hps.emplace_back(normal, 0);

    // create grid
    theta_1d = grid_t::make_grid(n_thetas_1d, lower, upper);
    dAryInt bits(n_thetas_1d, grid_dim);
    grid_range_t grid_range(grid_dim, bits.n_unique());
    for (size_t j = 0; j < bits.n_unique(); ++j, ++bits) {
        for (size_t i = 0; i < grid_dim; ++i) {
            grid_range.thetas()(i, j) = theta_1d[bits()(i)];
        }
    }
    grid_range.radii().array() = Gridder::radius(n_thetas_1d, lower, upper);
    grid_range.sim_sizes().array() = n_sim;

    grid_range.create_tiles(hps);
    grid_range.prune();

    bckt_t model(grid_dim, ph2_size, n_samples, thresholds);
    model.set_grid_range(grid_range);

    is_t is_o;

    for (auto _ : state) {
        fit<std::mt19937>(model, grid_range, is_o, n_sim, 0,
                          std::thread::hardware_concurrency());
    }
}

BENCHMARK_REGISTER_F(binomial_fixture, bench_fit);

}  // namespace kevlar
