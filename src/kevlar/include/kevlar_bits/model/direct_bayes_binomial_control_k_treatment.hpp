#pragma once
#include <algorithm>
#include <iostream>
#include <kevlar_bits/model/base.hpp>
#include <kevlar_bits/util/algorithm.hpp>
#include <kevlar_bits/util/legendre.hpp>
#include <kevlar_bits/util/macros.hpp>
#include <kevlar_bits/util/math.hpp>
#include <kevlar_bits/util/types.hpp>
#include <limits>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

class AllocCounter;

namespace kevlar {

class DirectBayesBinomialControlkTreatment {
   public:
    using vec_t = Eigen::VectorXd;
    using mat_t = Eigen::MatrixXd;

    static mat_t fast_invert(mat_t S, const vec_t &d) {
        for (int k = 0; k < d.size(); ++k) {
            auto offset = (d[k] / (1 + d[k] * S(k, k))) * (S.col(k) * S.row(k));
            S -= offset;
        }
        return S;
    }

    static vec_t conditional_exceed_prob_given_sigma(
        double sigma_sq, double mu_sig_sq, const vec_t &sample_I,
        const vec_t &MLE, const vec_t &thresholds, const vec_t &mu_0,
        const bool use_fast_inverse = false) {
        const int d = sample_I.size();
        mat_t S_0 = vec_t::Constant(d, sigma_sq).asDiagonal();
        S_0.array() += mu_sig_sq;

        // V_0 = solve(S_0)
        // but because this is a known case of the form
        // aI + bJ, we can use the explicit inverse formula, given by : 1 /
        // a I - J *(b / (a(a + db))) Note, by the way, that it's probably
        // possible to use significant precomputation here
        mat_t V_0 = vec_t::Constant(d, 1 / sigma_sq).asDiagonal();
        V_0.array() -= (mu_sig_sq / sigma_sq) / (sigma_sq + d * mu_sig_sq);
        mat_t Sigma_posterior;
        if (use_fast_inverse) {
            Sigma_posterior = fast_invert(S_0, sample_I);
        } else {
            mat_t precision_posterior = sample_I.asDiagonal();
            precision_posterior += V_0;
            Sigma_posterior =
                precision_posterior.llt().solve(mat_t::Identity(d, d));
        }

        const auto mu_posterior =
            Sigma_posterior *
            (sample_I.array() * MLE.array() + (V_0.matrix() * mu_0).array())
                .matrix();

        // TODO: Handle multiple thresholds like James does
        const auto z_scores = (mu_posterior.array() - thresholds[0]).array() /
                              Sigma_posterior.diagonal().array().sqrt();
        // James suggestion:
        // vec_t some_vec = (sample_I.array() * MLE.array() + (V_0.matrix() *
        // mu_0).array()).matrix(); vec_t z_scores = Sigma_posterior * some_vec;
        // z_scores.array() = (z_scores.array()-thresholds[0]) /
        // Sigma_posterior.diagonal().array().sqrt();
        return z_scores;
    }

    // let's evaluate the endpoints of the prior in logspace-sigma:
    // determine endpoints:
    static std::pair<vec_t, vec_t> get_quadrature(const double alpha_prior,
                                                  const double beta_prior,
                                                  const int n_points,
                                                  const int n_arm_size) {
        // Shared for a given prior

        // TODO: consider constexpr
        const double a = std::log(1e-8);
        const double b = std::log(1e3);
        auto pair = leggauss(n_points);
        // TODO: transpose this in leggauss for efficiency
        vec_t quadrature_points = pair.row(0);
        vec_t quadrature_weights = pair.row(1);
        quadrature_points =
            ((quadrature_points.array() + 1) * ((b - a) / 2) + a).exp();
        // sum(wts) = b-a so it averages to 1 over space
        quadrature_weights = quadrature_weights * ((b - a) / 2);
        // TODO: remove second alloc here
        vec_t density_logspace =
            invgamma_pdf(quadrature_points, alpha_prior, beta_prior);
        density_logspace.array() *= quadrature_points.array();
        auto weighted_density_logspace =
            density_logspace.array() * quadrature_weights.array();
        return {quadrature_points, weighted_density_logspace};
    }

    static vec_t get_false_rejections(const vec_t &phat,
                                      const vec_t &quadrature_points,
                                      const vec_t &weighted_density_logspace,
                                      const Eigen::Vector2d &thresholds,
                                      const int n_arm_size,
                                      const double mu_sig_sq) {
        // Shared for a given phat
        const int d = phat.size();
        const vec_t MLE = logit(phat);
        const vec_t sample_I = n_arm_size * phat.array() * (1 - phat.array());
        const int n_points = quadrature_points.size();

        mat_t exceed_probs(4, n_points);

        // TODO: make this user-specified
        const vec_t mu_0 = vec_t::Constant(d, 0);

        for (int i = 0; i < n_points; ++i) {
            exceed_probs.col(i) = conditional_exceed_prob_given_sigma(
                quadrature_points[i],
                mu_sig_sq,  // TODO: integrate over this too
                sample_I, MLE, thresholds, mu_0);
        }

        const auto posterior_exceedance_z_scores =
            (exceed_probs * weighted_density_logspace.matrix()).array() /
            (weighted_density_logspace.sum());
        return posterior_exceedance_z_scores;
    }
};

}  // namespace kevlar
