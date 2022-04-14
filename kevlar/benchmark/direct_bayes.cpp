#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <iostream>
#include <kevlar_bits/grid/grid_range.hpp>
#include <kevlar_bits/grid/gridder.hpp>
#include <kevlar_bits/grid/hyperplane.hpp>
#include <kevlar_bits/grid/tile.hpp>
#include <kevlar_bits/model/direct_bayes_binomial_control_k_treatment.hpp>
#include <kevlar_bits/util/macros.hpp>

namespace kevlar {
namespace {

using value_t = double;
using uint_t = uint32_t;
using tile_t = Tile<value_t>;
using gr_t = GridRange<value_t, uint_t, tile_t>;
using bckt_t = DirectBayesBinomialControlkTreatment<value_t>;
using vec_t = DirectBayesBinomialControlkTreatment<value_t>::vec_t;

const Eigen::Vector<value_t, 1> critical_values{0.95};
const auto phat = Eigen::Vector<value_t, 4>{28, 14, 33, 36}.array() / 50;
const value_t alpha_prior = 0.0005;
const value_t beta_prior = 0.000005;
const value_t mu_sig_sq = 0.1;
const int n_arm_size = 50;
const int n_arms = 4;
const int n_integration_points = 50;
const int n_thetas = 32;
const size_t n_samples = 250;
const value_t efficacy_threshold = 0.3;
const auto [quadrature_points, weighted_density_logspace] =
    DirectBayesBinomialControlkTreatment<value_t>::get_quadrature(
        alpha_prior, beta_prior, n_integration_points, n_arm_size);

vec_t get_efficacy_thresholds() {
    Eigen::Vector<value_t, Eigen::Dynamic> efficacy_thresholds(n_arms);
    efficacy_thresholds.fill(efficacy_threshold);
    return efficacy_thresholds;
}
struct MockHyperPlane : HyperPlane<value_t> {
    using base_t = HyperPlane<value_t>;
    using base_t::base_t;
};

static void BM_get_posterior_exceedance_probs(benchmark::State& state) {
    for (auto _ : state) {
        const auto got = DirectBayesBinomialControlkTreatment<
            value_t>::get_posterior_exceedance_probs(phat, quadrature_points,
                                                     weighted_density_logspace,
                                                     get_efficacy_thresholds(),
                                                     n_arm_size, mu_sig_sq);
    }
}

BENCHMARK(BM_get_posterior_exceedance_probs);

static void BM_rej_len(benchmark::State& state) {
    using hp_t = MockHyperPlane;
    auto theta_1d = Gridder::make_grid(n_thetas, -1., 0.);
    auto radius = Gridder::radius(n_thetas, -1., 0.);

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

    Eigen::Vector<value_t, Eigen::Dynamic> efficacy_thresholds(n_arms);
    efficacy_thresholds.fill(efficacy_threshold);
    bckt_t model(n_arms, n_samples, critical_values, efficacy_thresholds);
    model.set_grid_range(grid_range);
    size_t seed = 3214;
    std::mt19937 gen;
    gen.seed(seed);
    for (auto _ : state) {
        auto mstate = model.make_state();
        mstate->gen_rng(gen);
        mstate->gen_suff_stat();
        colvec_type<uint_t> rejs(grid_range.n_tiles());
        mstate->rej_len(rejs);
    }
}

BENCHMARK(BM_rej_len);

}  // namespace
}  // namespace kevlar