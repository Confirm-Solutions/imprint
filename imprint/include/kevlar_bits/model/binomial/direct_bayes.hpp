#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <imprint_bits/model/base.hpp>
#include <imprint_bits/model/binomial/common/fixed_n_default.hpp>
#include <imprint_bits/model/fixed_single_arm_size.hpp>
#include <imprint_bits/util/algorithm.hpp>
#include <imprint_bits/util/legendre.hpp>
#include <imprint_bits/util/macros.hpp>
#include <imprint_bits/util/math.hpp>
#include <imprint_bits/util/types.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

namespace imprint {
namespace model {
namespace binomial {

template <class ValueType>
struct DirectBayes : FixedSingleArmSize, ModelBase<ValueType> {
    using arm_base_t = FixedSingleArmSize;
    using base_t = ModelBase<ValueType>;
    using typename base_t::value_t;

   private:
    using vec_t = colvec_type<value_t>;
    using mat_t = mat_type<value_t>;

    static constexpr int n_integration_points = 16;
    static constexpr value_t alpha_prior = 0.0005;
    static constexpr value_t beta_prior = 0.000005;
    const vec_t efficacy_thresholds_;

   public:
    template <class _GenType, class _ValueType, class _UIntType,
              class _GridRangeType>
    struct SimGlobalState;

    template <class _GenType, class _ValueType, class _UIntType,
              class _GridRangeType>
    using sim_global_state_t =
        SimGlobalState<_GenType, _ValueType, _UIntType, _GridRangeType>;

    template <class _GridRangeType>
    using imprint_bound_state_t =
        ImprintBoundStateFixedNDefault<_GridRangeType>;

    DirectBayes(
        size_t n_arms, size_t n_arm_size,
        const Eigen::Ref<const colvec_type<value_t>>& cv,
        const Eigen::Ref<const colvec_type<value_t>>& efficacy_thresholds)
        : arm_base_t(n_arms, n_arm_size),
          base_t(),
          efficacy_thresholds_(efficacy_thresholds) {
        assert(efficacy_thresholds.size() == n_arms);
        critical_values(cv);
    }

    using arm_base_t::n_arm_samples;
    using arm_base_t::n_arms;

    using base_t::critical_values;
    void critical_values(const Eigen::Ref<const colvec_type<value_t>>& cv) {
        auto& cv_ = base_t::critical_values();
        cv_ = cv;
        std::sort(cv_.begin(), cv_.end(), std::greater<value_t>());
    }

    template <class _GenType, class _ValueType, class _UIntType,
              class _GridRangeType>
    auto make_sim_global_state(const _GridRangeType& grid_range) const {
        return sim_global_state_t<_GenType, _ValueType, _UIntType,
                                  _GridRangeType>(*this, grid_range);
    }

    template <class _GridRangeType>
    auto make_imprint_bound_state(const _GridRangeType& gr) const {
        return imprint_bound_state_t<_GridRangeType>(n_arm_samples(), gr);
    }
};

template <class ValueType>
template <class _GenType, class _ValueType, class _UIntType,
          class _GridRangeType>
struct DirectBayes<ValueType>::SimGlobalState
    : SimGlobalStateFixedNDefault<_GenType, _ValueType, _UIntType,
                                  _GridRangeType> {
    struct SimState;

    using base_t = SimGlobalStateFixedNDefault<_GenType, _ValueType, _UIntType,
                                               _GridRangeType>;
    using typename base_t::gen_t;
    using typename base_t::grid_range_t;
    using typename base_t::interface_t;
    using typename base_t::uint_t;
    using typename base_t::value_t;

    using sim_state_t = SimState;

   private:
    using model_t = DirectBayes;
    const model_t& model_;
    vec_t quadrature_points_;
    vec_t weighted_density_logspace_;
    Eigen::Tensor<Eigen::Vector<value_t, 4>, 4> posterior_exceedance_cache_;
    const double mu_sig_sq_ = 100;

   public:
    SimGlobalState(const model_t& model, const grid_range_t& grid_range)
        : base_t(model.n_arm_samples(), grid_range), model_(model) {
        const int n_arm_size = model_.n_arm_samples();
        const auto n_params = grid_range.n_params();
        std::tie(quadrature_points_, weighted_density_logspace_) =
            get_quadrature(model.alpha_prior, model.beta_prior,
                           model.n_integration_points, n_arm_size);
        // Under the current cache design, the number of arms must be known at
        // compile time
        assert(n_params == 4);
        posterior_exceedance_cache_.resize(n_arm_size, n_arm_size, n_arm_size,
                                           n_arm_size);
        vec_t suff_stats(n_params);
        posterior_exceedance_cache_.setConstant(
            Eigen::Vector<value_t, 4>::Constant(NAN));
        // Must start at 1 because DB is undefined at zero!
        for (int i = 1; i < n_arm_size - 1; ++i) {
            suff_stats[0] = i;
            for (int j = 1; j < n_arm_size - 1; ++j) {
                suff_stats[1] = j;
                for (int k = 1; k < n_arm_size - 1; ++k) {
                    suff_stats[2] = k;
                    for (int l = 1; l < n_arm_size - 1; ++l) {
                        suff_stats[3] = l;
                        posterior_exceedance_cache_(i, j, k, l) =
                            get_posterior_exceedance_probs(
                                suff_stats.array() / n_arm_size,
                                quadrature_points_, weighted_density_logspace_,
                                model_.efficacy_thresholds_,
                                model_.n_arm_samples(), mu_sig_sq_);
                    }
                }
            }
        }
    }

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
            invgamma_pdf(quadrature_points, alpha_prior, beta_prior)
                .template cast<value_t>();
        density_logspace.array() *= quadrature_points.array();
        auto weighted_density_logspace =
            density_logspace.array() * quadrature_weights.array();
        return {quadrature_points, weighted_density_logspace};
    }

    static mat_t faster_invert(const vec_t& D_inverse, value_t O) {
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
        const value_t sigma_sq, const value_t mu_sig_sq, const vec_t& sample_I,
        const vec_t& thetahat, const vec_t& logit_thresholds, const vec_t& mu_0,
        const bool use_fast_inverse = true) {
        const int d = sample_I.size();
        // TODO: precompute sigma_sq_inv, V_0, shift
        // TODO: minimize the heap allocations in this function
        vec_t sigma_sq_inv = vec_t::Constant(d, 1. / sigma_sq);
        mat_t V_0 = sigma_sq_inv.asDiagonal();
        auto shift = -1 * (mu_sig_sq / sigma_sq) / (sigma_sq + d * mu_sig_sq);
        V_0.array() += shift;
        mat_t Sigma_posterior;
        // TODO template this and use if constexpr
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

    static vec_t get_posterior_exceedance_probs(
        const vec_t& phat, const vec_t& quadrature_points,
        const vec_t& weighted_density_logspace,
        const vec_t& efficacy_thresholds, const size_t n_arm_size,
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
            // TODO: template this and use if constexpr
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

    std::unique_ptr<typename interface_t::sim_state_t> make_sim_state(
        size_t seed) const override {
        return std::make_unique<sim_state_t>(*this, seed);
    }
};

template <class ValueType>
template <class _GenType, class _ValueType, class _UIntType,
          class _GridRangeType>
struct DirectBayes<ValueType>::SimGlobalState<_GenType, _ValueType, _UIntType,
                                              _GridRangeType>::SimState
    : base_t::sim_state_t {
   private:
    using outer_t = SimGlobalState;

   public:
    using base_t = typename outer_t::base_t::sim_state_t;
    using typename base_t::interface_t;

   private:
    const outer_t& outer_;

   public:
    SimState(const outer_t& sgs, size_t seed)
        : base_t(sgs, seed), outer_(sgs) {}

    void simulate(Eigen::Ref<colvec_type<uint_t>> rej_len) override {
        base_t::generate_data();
        base_t::generate_sufficient_stats();

        const auto& bits = outer_.bits();
        const auto& gr_view = outer_.grid_range();

        const auto n_params = gr_view.n_params();  // same as n_arms
        const auto& critical_values = outer_.model_.critical_values();

        size_t pos = 0;

        for (size_t grid_i = 0; grid_i < gr_view.n_gridpts(); ++grid_i) {
            auto bits_i = bits.col(grid_i);

            Eigen::array<long, 4> suff_stats;
            for (int i = 0; i < n_params; ++i) {
                const auto& ss_i = base_t::sufficient_stats_arm(i);
                suff_stats[i] = ss_i(bits_i[i]);
            }
            const Eigen::Vector<value_t, 4>& posterior_exceedance_probs =
                outer_.posterior_exceedance_cache_(suff_stats);
            assert(posterior_exceedance_probs.array().sum() != 0);

            // assuming critical_values is sorted in descending order
            bool do_optimized_update =
                (posterior_exceedance_probs.array() <=
                 critical_values[critical_values.size() - 1])
                    .all();
            if (do_optimized_update) {
                rej_len.segment(pos, gr_view.n_tiles(grid_i)).array() = 0;
                pos += gr_view.n_tiles(grid_i);
                continue;
            }

            for (size_t n_t = 0; n_t < gr_view.n_tiles(grid_i); ++n_t, ++pos) {
                value_t max_null_prob_exceed = 0;
                for (int arm_i = 0; arm_i < n_params; ++arm_i) {
                    if (gr_view.check_null(pos, arm_i)) {
                        max_null_prob_exceed =
                            std::max(max_null_prob_exceed,
                                     posterior_exceedance_probs[arm_i]);
                    }
                }

                int cv_i = 0;
                for (; cv_i < critical_values.size(); ++cv_i) {
                    if (max_null_prob_exceed > critical_values[cv_i]) {
                        break;
                    }
                }
                rej_len(pos) = critical_values.size() - cv_i;
            }
        }

        assert(rej_len.size() == pos);
    }

    using base_t::score;
};

}  // namespace binomial
}  // namespace model
}  // namespace imprint
