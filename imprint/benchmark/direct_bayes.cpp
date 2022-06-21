#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <imprint_bits/bound/accumulator/typeI_error_accum.hpp>
#include <imprint_bits/driver/accumulate.hpp>
#include <imprint_bits/grid/grid_range.hpp>
#include <imprint_bits/grid/gridder.hpp>
#include <imprint_bits/grid/hyperplane.hpp>
#include <imprint_bits/grid/tile.hpp>
#include <imprint_bits/model/binomial/direct_bayes.hpp>
#include <imprint_bits/util/macros.hpp>
#include <iostream>
#include <random>

namespace imprint {
namespace {

using gen_t = std::mt19937;
using value_t = double;
using uint_t = uint32_t;
using tile_t = grid::Tile<value_t>;
using gr_t = grid::GridRange<value_t, uint_t, tile_t>;
using model_t = model::binomial::DirectBayes<value_t>;
using sgs_t =
    typename model_t::template sim_global_state_t<gen_t, value_t, uint_t, gr_t>;
using ss_t = typename sgs_t::sim_state_t;
using vec_t = colvec_type<value_t>;
using acc_t = bound::TypeIErrorAccum<value_t, uint_t>;

const Eigen::Vector<value_t, 1> critical_values{0.95};
const value_t alpha_prior = 0.0005;
const value_t beta_prior = 0.000005;
const value_t mu_sig_sq = 0.1;
const int n_integration_points = 16;
const int n_arm_size = 27;
const int n_arms = 4;
const int n_thetas = 64;
const size_t sim_size = 60 * 10;
const value_t efficacy_threshold = 0.3;
const auto [quadrature_points, weighted_density_logspace] =
    sgs_t::get_quadrature(alpha_prior, beta_prior, n_integration_points,
                          n_arm_size);

vec_t get_efficacy_thresholds() {
    Eigen::Vector<value_t, Eigen::Dynamic> efficacy_thresholds(n_arms);
    efficacy_thresholds.fill(efficacy_threshold);
    return efficacy_thresholds;
}

struct MockHyperPlane : grid::HyperPlane<value_t> {
    using base_t = HyperPlane<value_t>;
    using base_t::base_t;
};

static void BM_get_posterior_exceedance_probs(benchmark::State& state) {
    const auto phat = Eigen::Vector<value_t, 4>{28, 14, 33, 36}.array() / 50;
    for (auto _ : state) {
        const auto got = sgs_t::get_posterior_exceedance_probs(
            phat, quadrature_points, weighted_density_logspace,
            get_efficacy_thresholds(), n_arm_size, mu_sig_sq);
    }
}

BENCHMARK(BM_get_posterior_exceedance_probs);

static void BM_rej_len(benchmark::State& state) {
    using hp_t = MockHyperPlane;
    auto theta_1d = grid::Gridder::make_grid(n_thetas, -1., 0.);
    auto radius = grid::Gridder::radius(n_thetas, -1., 0.);

    colvec_type<value_t> normal(n_arms);
    std::vector<hp_t> hps;
    for (int i = 0; i < n_arms; ++i) {
        normal.setZero();
        normal(i) = -1;
        hps.emplace_back(normal, logit(efficacy_threshold));
    }

    // populate theta as the cartesian product of theta_1d
    gr_t grid_range(n_arms, ipow(n_thetas, n_arms));
    auto& thetas = grid_range.thetas();
    dAryInt bits(n_thetas, n_arms);
    for (size_t j = 0; j < grid_range.n_gridpts(); ++j) {
        for (size_t i = 0; i < n_arms; ++i) {
            thetas(i, j) = theta_1d[bits()[i]];
        }
        ++bits;
    }

    // populate radii as fixed radius
    grid_range.radii().array() = radius;

    // create tile information
    grid_range.create_tiles(hps);
    grid_range.prune();

    colvec_type<value_t> efficacy_thresholds(n_arms);
    efficacy_thresholds.fill(efficacy_threshold);
    size_t n_threads = std::thread::hardware_concurrency();
    model_t model(n_arms, n_arm_size, critical_values, efficacy_thresholds);

    auto sgs =
        model.make_sim_global_state<gen_t, value_t, uint_t, gr_t>(grid_range);
    size_t seed = 3214;
    colvec_type<uint_t> rej_len(grid_range.n_tiles());
    acc_t acc_os(critical_values.size(), grid_range.n_tiles(), n_arms);
    for (auto _ : state) {
        driver::accumulate(sgs, grid_range, acc_os, sim_size, seed, n_threads);
    }
}

BENCHMARK(BM_rej_len);

}  // namespace
}  // namespace imprint
