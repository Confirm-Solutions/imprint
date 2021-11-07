#include <kevlar_bits/model/binomial_control_k_treatment.hpp>
#include <kevlar_bits/util/algorithm.hpp>
#include <kevlar_bits/util/grid.hpp>
#include <kevlar_bits/process/tune.hpp>
#include "benchmark/benchmark.h"

namespace kevlar {

struct binomial_fixture : benchmark::Fixture
{
    using grid_t = Gridder<grid::Rectangular>;
    size_t p_size = 32;
    double lower = -0.5;
    double upper = 1.5;
    size_t n_sim = 100;
    double alpha = 0.05;
    double delta = 0.05;
    size_t grid_dim = 3;
    size_t grid_radius = grid_t::radius(p_size, lower, upper);
    size_t n_samples = 250;
    size_t ph2_size = 50;
    Eigen::VectorXd p_1d;
    Eigen::MatrixXd p_endpt;
    Eigen::VectorXd thr_vec;
};

BENCHMARK_DEFINE_F(binomial_fixture,
                   bench_tune)(benchmark::State& state)
{ 
    const size_t lmda_size = state.range(0);
    const size_t batch_size = state.range(1);

    p_1d = grid_t::make_grid(p_size, lower, upper);
    p_1d = p_1d.unaryExpr([](auto x) { return 1./(1. + std::exp(-x)); });

    p_endpt = grid_t::make_endpts(p_size, lower, upper);
    p_endpt = p_endpt.unaryExpr([](auto x) { return 1./(1. + std::exp(-x)); });

    thr_vec = grid_t::make_grid(lmda_size, 14., 15.2);
    sort_cols(thr_vec, std::greater<double>());

    auto rng_gen_f = [=](auto& gen, auto& rng) {
        std::uniform_real_distribution<double> unif(0., 1.);
        rng = Eigen::MatrixXd::NullaryExpr(n_samples, grid_dim, 
                [&](auto, auto) { return unif(gen); });
    };

    BinomialControlkTreatment<grid::Rectangular> 
        model(grid_dim, ph2_size, n_samples);

    for (auto _ : state) {
        state.PauseTiming();
        state.ResumeTiming();

        try {
            auto thr = tune(n_sim, alpha, delta, grid_dim, grid_radius,
                 p_1d, p_endpt, thr_vec, rng_gen_f, model, 0, batch_size, 
                 ProgressBarOSWrapper<void_ostream>() );
            benchmark::DoNotOptimize(thr);
        } 
        catch (const kevlar_error& e) {
            std::cerr << e.what() << std::endl;
        }
    }

}

BENCHMARK_REGISTER_F(binomial_fixture,
                     bench_tune)
    -> ArgsProduct({
            {30},
            {}
        })
    ;

} // namespace kevlar
