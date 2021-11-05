#include <kevlar_bits/sampler/thompson.hpp>
#include <kevlar_bits/distribution/beta.hpp>
#include <kevlar_bits/util/d_ary_int.hpp>
#include <kevlar_bits/process/tune.hpp>
#include <iostream>

int main() {
    using namespace kevlar;

    Eigen::VectorXd p_1d(128);
    p_1d.setRandom();
    p_1d.array() = (p_1d.array() / 6. + 0.5);
    std::sort(p_1d.data(), p_1d.data() + p_1d.size());
    Eigen::MatrixXd p(128 * 128, 2);
    dAryInt idx(p_1d.size(), p.cols());
    for (int i = 0; i < p.rows(); ++i, ++idx) {
        auto& bits = idx();
        for (int j = 0; j < p.cols(); ++j) {
            p(i, j) = p_1d(bits(j));
        }
    }

    double alpha = 0.05;
    double delta = 0.05;
    size_t grid_dim = 2;
    size_t grid_radius =

        tune(
            100, alpha, []() { return Eigen::VectorXd(); },
            [&](const auto&) {
                auto J = p.rows();
                auto d = p.cols();
                Thompson<double> thompson;
                Beta<double> beta;
                double alpha_prior = 1;
                double beta_prior = 1;
                int max_patients = 100;
                Eigen::MatrixXi n_action_arms(J, d);
                n_action_arms.setZero();
                Eigen::MatrixXi successes(J, d);
                successes.setZero();
                Eigen::MatrixXd beta_posterior(J, d);
                beta_posterior.setZero();
                Eigen::VectorXd alpha_posterior_buff(J * d);
                alpha_posterior_buff.setZero();
                Eigen::Map<Eigen::MatrixXd> alpha_posterior(
                    alpha_posterior_buff.data(), J, d);

                thompson.sample(p, beta, alpha_prior, beta_prior, max_patients,
                                n_action_arms, successes, alpha_posterior,
                                beta_posterior);
                Eigen::VectorXd out =
                    alpha_posterior.col(alpha_posterior.cols() - 1);
                return out;
            });

    return 0;
}
