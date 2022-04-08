#pragma once
#include <algorithm>
#include <Eigen/Dense>
#include <iostream>
#include <kevlar_bits/grid/tile.hpp>
#include <kevlar_bits/model/base.hpp>
#include <kevlar_bits/model/binomial_control_k_treatment.hpp>
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

namespace kevlar {

// TODO: template UIntType and GridRangeType
template <class ValueType = double>
class DirectBayesBinomialControlkTreatment
    : public BinomialControlkTreatment<
          ValueType, uint32_t,
          GridRange<ValueType, uint32_t, Tile<ValueType>>> {
   public:
    using value_t = ValueType;
    using uint_t = uint32_t;
    using base_t = BinomialControlkTreatment<
        ValueType, uint32_t, GridRange<ValueType, uint32_t, Tile<ValueType>>>;
    using vec_t = typename base_t::vec_t;
    using mat_t = typename base_t::mat_t;
    using base_t::make_state;
    using base_t::set_grid_range;

   private:
    vec_t quadrature_points_;
    vec_t weighted_density_logspace_;
    const vec_t efficacy_thresholds_;

   public:
    DirectBayesBinomialControlkTreatment(
        size_t n_arms, size_t n_arm_size,
        const Eigen::Ref<const colvec_type<ValueType>> &critical_values,
        const vec_t &efficacy_thresholds)
        : base_t(n_arms, 1, n_arm_size, critical_values),
          efficacy_thresholds_(efficacy_thresholds) {
        const int n_integration_points = 50;
        const double alpha_prior = 0.0005;
        const double beta_prior = 0.000005;
        assert(efficacy_thresholds.size() == n_arms);
        std::tie(quadrature_points_, weighted_density_logspace_) =
            DirectBayesBinomialControlkTreatment<value_t>::get_quadrature(
                alpha_prior, beta_prior, n_integration_points, n_arm_size);
    }

    static mat_t faster_invert(const vec_t &D_inverse, const value_t O) {
        //(1) compute multiplier on the new rank-one component
        auto multiplier = -O / (1 + O * D_inverse.sum());
        mat_t M = multiplier * D_inverse * D_inverse.transpose();
        M.diagonal() += D_inverse;
        return M;
    }

    static value_t faster_determinant(const vec_t D_inverse, const value_t O) {
        // This function uses "Sherman-Morrison for determinants"
        // https://en.wikipedia.org/wiki/Matrix_determinant_lemma
        // Note: this can be embedded inside of faster_invert to take advantage
        // of partial existing computations. If only I knew how to coveniently
        // return multiple objects...lol
        auto detD_inverse = (1. / D_inverse.array()).prod();
        auto newdeterminant = detD_inverse * (1 + O * D_inverse.sum());
        return newdeterminant;
    }

    static vec_t conditional_exceed_prob_given_sigma(
        value_t sigma_sq, value_t mu_sig_sq, const vec_t &sample_I,
        const vec_t &thetahat, const vec_t &logit_thresholds, const vec_t &mu_0,
        const bool use_fast_inverse = true) {
        const int d = sample_I.size();
        mat_t S_0 = vec_t::Constant(d, sigma_sq).asDiagonal();
        S_0.array() += mu_sig_sq;
        ASSERT_GOOD(S_0);

        vec_t sigma_sq_inv = vec_t::Constant(d, 1. / sigma_sq);
        mat_t V_0 = sigma_sq_inv.asDiagonal();
        auto shift = -1 * (mu_sig_sq / sigma_sq) / (sigma_sq + d * mu_sig_sq);
        V_0.array() += shift;
        mat_t Sigma_posterior;
        if (use_fast_inverse) {
            vec_t V_0 = 1. / (sigma_sq_inv + sample_I).array();
            Sigma_posterior = faster_invert(V_0, shift);
        } else {
            mat_t precision_posterior = sample_I.asDiagonal();
            precision_posterior += V_0;
            Sigma_posterior =
                precision_posterior.llt().solve(mat_t::Identity(d, d));
        }
        ASSERT_GOOD(Sigma_posterior);

        ASSERT_GOOD(sample_I);
        ASSERT_GOOD(thetahat);
        const vec_t mu_posterior =
            Sigma_posterior *
            (sample_I.array() * thetahat.array() + (V_0 * mu_0).array())
                .matrix();
        ASSERT_GOOD(mu_posterior);

        vec_t z_scores = (mu_posterior - logit_thresholds).array();
        z_scores.array() /= Sigma_posterior.diagonal().array().sqrt();
        // James suggestion:
        // vec_t some_vec = (sample_I.array() * thetahat.array() +
        // (V_0.matrix()
        // * mu_0).array()).matrix(); vec_t z_scores = Sigma_posterior *
        // some_vec; z_scores.array() = (z_scores.array()-thresholds[0]) /
        // Sigma_posterior.diagonal().array().sqrt();
        ASSERT_GOOD(z_scores);
        return normal_cdf(z_scores);
    }

    // let's evaluate the endpoints of the prior in logspace-sigma:
    // determine endpoints:
    static std::pair<vec_t, vec_t> get_quadrature(
        const value_t alpha_prior, const value_t beta_prior,
        const int n_integration_points, const int n_arm_size) {
        // Shared for a given prior
        // TODO: consider constexpr
        const value_t a = std::log(1e-8);
        const value_t b = std::log(1e3);
        auto pair = leggauss(n_integration_points);
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

    static vec_t get_posterior_exceedance_probs(
        const vec_t &phat, const vec_t &quadrature_points,
        const vec_t &weighted_density_logspace,
        const vec_t &efficacy_thresholds, const size_t n_arm_size,
        const value_t mu_sig_sq, bool use_optimized = true) {
        assert((phat.array() >= 0).all());
        assert((phat.array() <= 1).all());
        // Shared for a given thetahat
        const int n_arms = phat.size();
        const vec_t thetahat = logit(phat.array());
        ASSERT_GOOD(thetahat);
        const vec_t sample_I = n_arm_size * phat.array() * (1 - phat.array());
        const int n_integration_points = quadrature_points.size();

        // TODO: make this a user-specified parameter
        const vec_t mu_0 = vec_t::Constant(n_arms, -1.34);

        vec_t sample_I_inv = 1.0 / sample_I.array();
        vec_t posterior_reweight(n_integration_points);
        for (int i = 0; i < n_integration_points; ++i) {
            if (use_optimized) {
                auto sigma_sq = quadrature_points[i];
                vec_t diaginv = 1.0 / (sample_I_inv.array() + sigma_sq);
                auto totalvar_inv = faster_invert(diaginv, mu_sig_sq);
                auto meandiff = thetahat - mu_0;
                auto exponent =
                    -0.5 *
                    (meandiff.transpose().dot((totalvar_inv * meandiff)));
                auto determinant_piece =
                    1. / std::sqrt(faster_determinant(diaginv, mu_sig_sq));
                posterior_reweight(i) = determinant_piece * std::exp(exponent);
            } else {
                mat_t total_var =
                    (vec_t::Constant(n_arms, quadrature_points[i]) +
                     sample_I_inv)
                        .asDiagonal();
                total_var.array() += mu_sig_sq;
                auto determinant = total_var.determinant();
                posterior_reweight(i) =
                    1. / std::sqrt(determinant) *
                    std::exp(-0.5 * (((thetahat - mu_0).transpose() *
                                      total_var.inverse()) *
                                     (thetahat - mu_0))
                                        .sum());
            }
        }
        vec_t final_reweight =
            (posterior_reweight.array() * weighted_density_logspace.array());
        final_reweight /= final_reweight.sum();

        const vec_t logit_efficacy_thresholds =
            logit(efficacy_thresholds.array());
        mat_t exceed_probs(n_arms, n_integration_points);
        for (int i = 0; i < n_integration_points; ++i) {
            exceed_probs.col(i) = conditional_exceed_prob_given_sigma(
                quadrature_points[i],
                mu_sig_sq,  // TODO: integrate over this too
                sample_I, thetahat, logit_efficacy_thresholds, mu_0);
        }

        const auto posterior_exceedance_probs =
            exceed_probs * final_reweight.matrix();
        return posterior_exceedance_probs;
    }

    struct StateType : public base_t::StateType {
       private:
        KEVLAR_STRONG_INLINE auto get_ss(int arm_i) {
            Eigen::Map<const colvec_type<uint_t>> map(
                this->suff_stat().data() + this->outer().strides()[arm_i],
                this->outer().strides()[arm_i + 1] -
                    this->outer().strides()[arm_i]);
            return map;
        }

        using outer_t = DirectBayesBinomialControlkTreatment;
        using model_state_base_t = typename base_t::StateType;

       public:
        const outer_t &outer() {
            return static_cast<const outer_t &>(base_t::StateType::outer());
        }

        StateType(const outer_t &t) : model_state_base_t(t){};
        virtual void rej_len(Eigen::Ref<colvec_type<uint_t>> rej_len) override {
            const auto &bits = this->outer().gbits();
            const auto &gr_view = this->outer().grid_range();
            const auto n_arms = this->outer().n_arms();
            const auto &critical_values = this->outer().thresholds();
            const double mu_sig_sq = 100;

            int pos = 0;
            for (int grid_i = 0; grid_i < this->outer().n_gridpts(); ++grid_i) {
                auto bits_i = bits.col(grid_i);

                vec_t phat(n_arms);
                for (int i = 0; i < n_arms; ++i) {
                    phat(i) = static_cast<value_t>(get_ss(i)(bits_i[i])) /
                              this->outer().n_samples();
                }
                vec_t posterior_exceedance_probs =
                    get_posterior_exceedance_probs(
                        phat, this->outer().quadrature_points_,
                        this->outer().weighted_density_logspace_,
                        this->outer().efficacy_thresholds_,
                        this->outer().n_samples(), mu_sig_sq);

                // assuming critical_values is sorted in descending order
                bool do_optimized_update =
                    (posterior_exceedance_probs.array() <=
                     critical_values[critical_values.size() - 1])
                        .all();
                PRINT(phat);
                PRINT(posterior_exceedance_probs);
                if (do_optimized_update) {
                    rej_len.segment(pos, gr_view.n_tiles(grid_i)).array() = 0;
                    pos += gr_view.n_tiles(grid_i);
                    continue;
                }

                for (size_t n_t = 0; n_t < gr_view.n_tiles(grid_i);
                     ++n_t, ++pos) {
                    value_t max_null_prob_exceed = 0;
                    for (int arm_i = 0; arm_i < n_arms; ++arm_i) {
                        if (gr_view.check_null(pos, arm_i)) {
                            max_null_prob_exceed =
                                std::max(max_null_prob_exceed,
                                         posterior_exceedance_probs[arm_i]);
                        }
                    }

                    auto it = std::find_if(
                        critical_values.begin(), critical_values.end(),
                        [&](auto t) { return max_null_prob_exceed > t; });
                    rej_len(pos) = std::distance(it, critical_values.end());
                }
            }
        }
    };
    using state_t = StateType;
    std::unique_ptr<typename base_t::StateType::model_state_base_t> make_state()
        const override {
        return std::make_unique<state_t>(*this);
    }
};

}  // namespace kevlar
