#pragma once
#include <Eigen/Core>
#include <random>

namespace kevlar {

/*
 * For now, it is defined to only sample for bernoulli responses,
 * beta (conjugate) prior, and with rng savings.
 */
template <class ValueType>
struct Thompson {
    using value_t = ValueType;
    using int_t = int;

    template <class PMatType, class PosteriorDistType, class NActionArmsType,
              class SuccessesType, class AlphaPosteriorType,
              class BetaPosteriorType, class GenType = std::mt19937>
    void sample(PMatType& p, PosteriorDistType post_dist, value_t alpha_prior,
                value_t beta_prior, int_t max_iter,
                NActionArmsType& n_action_arms, SuccessesType& successes,
                AlphaPosteriorType& alpha_posterior,
                BetaPosteriorType& beta_posterior,
                GenType&& gen = std::mt19937()) const {
        auto d = p.cols();
        auto J = p.rows();  // number of grid points

        // currenty only supports 2 arms
        static constexpr int n_arms_supported = 2;
        assert(d == n_arms_supported);
        assert(n_action_arms.cols() == n_arms_supported);
        assert(successes.cols() == n_arms_supported);
        assert(alpha_posterior.cols() == n_arms_supported);
        assert(beta_posterior.cols() == n_arms_supported);

        // initialize variables
        Eigen::Matrix<bool, Eigen::Dynamic, 1> actions(J);
        std::uniform_real_distribution<value_t> unif(0., 1.);
        Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic> posterior(J, d);

        // iterate through each patient
        for (int_t i = 0; i < max_iter; ++i) {
            for (int_t j = 0; j < n_arms_supported; ++j) {
                post_dist(n_action_arms.col(j), successes.col(j), alpha_prior,
                          beta_prior, posterior.col(j), gen);
            }

            value_t u = unif(gen);
            actions =
                (posterior.col(1).array() > posterior.col(0).array()).matrix();

            for (int_t j = 0; j < n_arms_supported; ++j) {
                successes.col(j).array() +=
                    ((actions.array() == j) && (p.col(j).array() > u))
                        .template cast<int>();
            }

            for (int_t j = 0; j < n_arms_supported; ++j) {
                n_action_arms.col(j).array() +=
                    (actions.array() == j).template cast<int>();
            }
        }

        alpha_posterior.array() +=
            successes.template cast<value_t>().array() + alpha_prior;
        beta_posterior.array() +=
            (n_action_arms - successes).template cast<value_t>().array() +
            beta_prior;
    }
};

}  // namespace kevlar
