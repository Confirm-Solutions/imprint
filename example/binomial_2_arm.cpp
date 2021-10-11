#include <kevlar_bits/sampler/thompson.hpp>
#include <kevlar_bits/distribution/beta.hpp>
#include <iostream>

int main()
{
    using namespace kevlar;

    Eigen::MatrixXd p(16, 2);
    p <<
    0.4378235, 0.4378235, 
    0.5621765, 0.4378235,
    0.6791787, 0.4378235,
    0.7772999, 0.4378235,
    0.4378235, 0.5621765,
    0.5621765, 0.5621765,
    0.6791787, 0.5621765,
    0.7772999, 0.5621765,
    0.4378235, 0.6791787,
    0.5621765, 0.6791787,
    0.6791787, 0.6791787,
    0.7772999, 0.6791787,
    0.4378235, 0.7772999,
    0.5621765, 0.7772999,
    0.6791787, 0.7772999,
    0.7772999, 0.7772999
        ;

    auto J = p.rows();
    auto d = p.cols();
    Thompson<double> thompson;
    Beta<double> beta;
    double alpha_prior = 1;
    double beta_prior = 1;
    int max_patients = 100;
    Eigen::MatrixXi n_action_arms(J, d);
    Eigen::MatrixXi successes(J, d);
    Eigen::MatrixXd alpha_posterior(J, d);
    Eigen::MatrixXd beta_posterior(J, d);

    thompson.sample(p, beta, alpha_prior, beta_prior, max_patients,
                    n_action_arms, successes,
                    alpha_posterior, beta_posterior);
    std::cout << alpha_posterior << std::endl;

    return 0;
}
