#pragma once
#include <random>
#include <Eigen/Core>

namespace kevlar {

template <class ValueType>
struct Beta {
    using value_t = ValueType;

    /*
     * TODO: needs clean-up!
     * Lots of assumptions about what this function does.
     */
    template <class NType, class SuccessType, class PosteriorType,
              class GenType = std::mt19937>
    void operator()(const NType& n, const SuccessType& s, value_t alpha_prior,
                    value_t beta_prior, PosteriorType&& post,
                    GenType&& gen = std::mt19937()) {
        std::gamma_distribution<value_t> gamma_a(alpha_prior);
        std::gamma_distribution<value_t> gamma_b(beta_prior);

        auto end_left = gamma_a(gen);
        auto end_right = gamma_b(gen);
        auto n_max = n.maxCoeff();
        g_sums_.resize(n_max + 1);
        g_sums_(0) = 0;
        for (int i = 1; i < g_sums_.size(); ++i) {
            g_sums_(i) = g_sums_(i - 1) + gamma_1_(gen);
        }

        for (int i = 0; i < n.size(); ++i) {
            post(i) = (g_sums_(s(i)) + end_left) /
                      (g_sums_(n(i)) + end_left + end_right);
        }
    }

   private:
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    std::gamma_distribution<value_t> gamma_1_;
    vec_t g_sums_;
};

}  // namespace kevlar
