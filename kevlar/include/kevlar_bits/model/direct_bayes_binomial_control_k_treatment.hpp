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
    : BinomialControlkTreatment<
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

    DirectBayesBinomialControlkTreatment(
        size_t n_arms, size_t n_samples,
        const Eigen::Ref<const colvec_type<ValueType>> &thresholds)
        : base_t(n_arms, 0, n_samples, thresholds) {}

    static mat_t fast_invert(mat_t S, const vec_t &d) {
        for (int k = 0; k < d.size(); ++k) {
            auto offset = (d[k] / (1 + d[k] * S(k, k))) * (S.col(k) * S.row(k));
            S -= offset;
        }
        return S;
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
            (sample_I.array() * thetahat.array() + (V_0 * mu_0).array())
                .matrix();

        mat_t z_scores(mu_posterior.size(), logit_thresholds.size());
        for (int i = 0; i < logit_thresholds.size(); ++i) {
            z_scores.col(i) = (mu_posterior.array() - logit_thresholds(i)) /
                              Sigma_posterior.diagonal().array().sqrt();
        }
        // James suggestion:
        // vec_t some_vec = (sample_I.array() * thetahat.array() +
        // (V_0.matrix()
        // * mu_0).array()).matrix(); vec_t z_scores = Sigma_posterior *
        // some_vec; z_scores.array() = (z_scores.array()-thresholds[0]) /
        // Sigma_posterior.diagonal().array().sqrt();
        return normal_cdf(z_scores);
    }

    // let's evaluate the endpoints of the prior in logspace-sigma:
    // determine endpoints:
    static std::pair<vec_t, vec_t> get_quadrature(const value_t alpha_prior,
                                                  const value_t beta_prior,
                                                  const int n_points,
                                                  const int n_arm_size) {
        // Shared for a given prior
        // TODO: consider constexpr
        const value_t a = std::log(1e-8);
        const value_t b = std::log(1e3);
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
                                      const vec_t &thresholds,
                                      const size_t n_arm_size,
                                      const value_t mu_sig_sq,
                                      bool use_optimized = true) {
        // Shared for a given phat
        const int d = phat.size();
        const vec_t thetahat = logit(phat);
        const vec_t sample_I = n_arm_size * phat.array() * (1 - phat.array());
        const int n_points = quadrature_points.size();

        // TODO: make this a user-specified parameter
        const vec_t mu_0 = vec_t::Constant(d, -1.34);

        vec_t sample_I_inv = 1.0 / sample_I.array();
        vec_t posterior_reweight(n_points);
        for (int i = 0; i < n_points; ++i) {
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
                    (vec_t::Constant(d, quadrature_points[i]) + sample_I_inv)
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

        const vec_t logit_thresholds = logit(thresholds);
        mat_t exceed_probs(4, n_points);
        for (int i = 0; i < n_points; ++i) {
            exceed_probs.col(i) = conditional_exceed_prob_given_sigma(
                quadrature_points[i],
                mu_sig_sq,  // TODO: integrate over this too
                sample_I, thetahat, logit_thresholds, mu_0);
        }

        const auto posterior_exceedance_z_scores =
            exceed_probs * final_reweight.matrix();
        return posterior_exceedance_z_scores;
    }

    struct StateType : base_t::StateType {
       private:
        template <class BitsType>
        KEVLAR_STRONG_INLINE auto rej_len_internal(size_t a_star,
                                                   BitsType &bits_i) {
            // pairwise z-test
            auto n = this->outer_.n_samples();
            Eigen::Map<const colvec_type<uint_t>> ss_astar(
                this->suff_stat_.data() + this->outer_.strides_[a_star],
                this->outer_.strides_[a_star + 1] -
                    this->outer_.strides_[a_star]);
            Eigen::Map<const colvec_type<uint_t>> ss_0(
                this->suff_stat_.data(), this->outer_.strides_[1]);
            auto p_star = static_cast<ValueType>(ss_astar(bits_i[a_star])) / n;
            auto p_0 = static_cast<ValueType>(ss_0(bits_i[0])) / n;
            auto z = (p_star - p_0);
            auto var = (p_star * (1. - p_star) + p_0 * (1. - p_0));
            z = (var <= 0) ? std::copysign(1.0, z) *
                                 std::numeric_limits<ValueType>::infinity()
                           : z / std::sqrt(var / n);

            int i = 0;
            for (; i < this->outer_.thresholds_.size(); ++i) {
                if (z > this->outer_.thresholds_[i]) break;
            }

            return this->outer_.n_models() - i;
        }

       protected:
        using outer_t = DirectBayesBinomialControlkTreatment;

       public:
        using model_state_base_t = typename base_t::StateType;
        StateType(const outer_t &t) : model_state_base_t(t){};
        virtual void rej_len(Eigen::Ref<colvec_type<uint_t>> rej_len) override {
            PRINT("rej_len");
            const auto &bits = this->outer().gbits();
            PRINT(bits);
            const auto &gr_view = this->outer().grid_range();

            int pos = 0;

            // TODO return number of rejected hypotheses at each threshold

            // for (int i = 0; i < outer_.n_gridpts(); ++i) {
            //     auto bits_i = bits.col(i);

            //     vec_t counts(bits_i.size());
            //     for (int j = 0; j < bits_i.size(); ++j) {
            //         Eigen::Map<const colvec_type<uint_t>> ph2_counts_v(
            //             ph2_counts_.data() + outer_.strides_[j] -
            //                 outer_.strides_[1],
            //             outer_.strides_[j + 1] - outer_.strides_[j]);
            //         counts[j] =
            //         static_cast<int64_t>(ph2_counts_v(bits_i[j]));
            //     }
            //     PRINT(counts);

            //     // Phase III

            // size_t rej = 0;

            // // if current gridpt is regular, do an optimized routine.
            // if (gr_view.is_regular(i)) {
            //     if (gr_view.check_null(pos, a_star - 1)) {
            //         rej = phase_III_internal(a_star, bits_i);
            //     }
            //     rej_len[pos] = rej;
            //     ++pos;
            //     continue;
            // }

            // // else, do a slightly different routine:
            // // compute the ph3 test statistic first and loop through
            // // each tile to check if it's a false rejection.
            // bool rej_computed = false;
            // const auto n_ts = gr_view.n_tiles(i);
            // for (size_t n_t = 0; n_t < n_ts; ++n_t, ++pos) {
            //     bool is_null = gr_view.check_null(pos, a_star - 1);
            //     if (!rej_computed && is_null) {
            //         rej = phase_III_internal(a_star, bits_i);
            //         rej_computed = true;
            //     }
            //     rej_len[pos] = is_null ? rej : 0;
            // }
        }
    };
    using state_t = StateType;
    std::unique_ptr<typename base_t::StateType::model_state_base_t> make_state()
        const override {
        return std::make_unique<state_t>(*this);
    }
};

}  // namespace kevlar
