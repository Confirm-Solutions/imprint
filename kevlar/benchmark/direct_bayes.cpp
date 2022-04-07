#include <benchmark/benchmark.h>
#include <kevlar_bits/model/direct_bayes_binomial_control_k_treatment.hpp>

namespace kevlar {
namespace {

const double mu_sig_sq = 0.1;
const double alpha_prior = 0.0005;
const double beta_prior = 0.000005;
const int n_integration_points = 50;
const int n_arm_size = 50;
const Eigen::Vector2d thresholds{1.96, 1.96};
const auto [quadrature_points, weighted_density_logspace] =
    DirectBayesBinomialControlkTreatment<>::get_quadrature(
        alpha_prior, beta_prior, n_integration_points, n_arm_size);
const auto phat = Eigen::Vector4d{28, 14, 33, 36}.array() / 50;

static void BM_get_posterior_exceedance_probs(benchmark::State& state) {
    for (auto _ : state) {
        const auto got = DirectBayesBinomialControlkTreatment<>::
            get_posterior_exceedance_probs(phat, quadrature_points,
                                           weighted_density_logspace,
                                           thresholds, n_arm_size, mu_sig_sq);
    }
}

BENCHMARK(BM_get_posterior_exceedance_probs);

static void BM_rej_len(benchmark::State& state) {
    auto model = get_test_class();
    auto grid_range = get_grid_range(64, 4);
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