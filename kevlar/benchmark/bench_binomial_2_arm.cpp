#include <benchmark/benchmark.h>
#include <kevlar_bits/sampler/thompson.hpp>
#include <kevlar_bits/distribution/beta.hpp>

namespace kevlar {

struct binomial_fixture : benchmark::Fixture {};

BENCHMARK_DEFINE_F(binomial_fixture, binomial)(benchmark::State& state) {
    Eigen::MatrixXd p(16, 2);
    p << 0.4378235, 0.4378235, 0.5621765, 0.4378235, 0.6791787, 0.4378235,
        0.7772999, 0.4378235, 0.4378235, 0.5621765, 0.5621765, 0.5621765,
        0.6791787, 0.5621765, 0.7772999, 0.5621765, 0.4378235, 0.6791787,
        0.5621765, 0.6791787, 0.6791787, 0.6791787, 0.7772999, 0.6791787,
        0.4378235, 0.7772999, 0.5621765, 0.7772999, 0.6791787, 0.7772999,
        0.7772999, 0.7772999;

    auto J = p.rows();
    auto d = p.cols();
    Thompson<double> thompson;
    Beta<double> beta;
    const double alpha_prior = 1;
    const double beta_prior = 1;
    const int max_patients = 100;
    Eigen::MatrixXi n_action_arms(J, d);
    Eigen::MatrixXi successes(J, d);
    Eigen::MatrixXd alpha_posterior(J, d);
    Eigen::MatrixXd beta_posterior(J, d);

    for (auto _ : state) {
        state.PauseTiming();
        n_action_arms.setZero();
        successes.setZero();
        alpha_posterior.setZero();
        beta_posterior.setZero();
        state.ResumeTiming();

        thompson.sample(p, beta, alpha_prior, beta_prior, max_patients,
                        n_action_arms, successes, alpha_posterior,
                        beta_posterior);
    }
}

BENCHMARK_REGISTER_F(binomial_fixture, binomial);

}  // namespace kevlar
