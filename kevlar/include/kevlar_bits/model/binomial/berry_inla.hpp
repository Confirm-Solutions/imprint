#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <kevlar_bits/model/base.hpp>
#include <kevlar_bits/model/binomial/common/fixed_n_default.hpp>
#include <kevlar_bits/model/fixed_single_arm_size.hpp>
#include <kevlar_bits/util/algorithm.hpp>
#include <kevlar_bits/util/legendre.hpp>
#include <kevlar_bits/util/macros.hpp>
#include <kevlar_bits/util/math.hpp>
#include <kevlar_bits/util/types.hpp>

namespace kevlar {
namespace model {
namespace binomial {

template <class ValueType, int ARMS>
struct BerryINLA : FixedSingleArmSize, ModelBase<ValueType> {
    using arm_base_t = FixedSingleArmSize;
    using base_t = ModelBase<ValueType>;
    using typename base_t::value_t;

   private:
    using vec_t = colvec_type<value_t>;
    using mat_t = mat_type<value_t>;

    static constexpr double mu_0 = -1.34;

    const size_t n_arm_samples;
    const vec_t efficacy_thresholds;

    // variables dependent solely on the quadrature points (independent of the
    // data).
    const vec_t quad_wts;
    const mat_t cov;
    const mat_t neg_precQ;
    const vec_t logprecQdet;
    const vec_t logprior;
    const value_t opt_tol2;
    const value_t logit_p1;

   public:
    template <class _GenType, class _ValueType, class _UIntType,
              class _GridRangeType>
    struct SimGlobalState;

    template <class _GenType, class _ValueType, class _UIntType,
              class _GridRangeType>
    using sim_global_state_t =
        SimGlobalState<_GenType, _ValueType, _UIntType, _GridRangeType>;

    template <class _GridRangeType>
    using kevlar_bound_state_t = KevlarBoundStateFixedNDefault<_GridRangeType>;

    BerryINLA(size_t n_arm_samples, const Eigen::Ref<const vec_t>& cv,
              const Eigen::Ref<const vec_t>& efficacy_thresholds,
              const Eigen::Ref<const vec_t>& quad_wts,
              const Eigen::Ref<const mat_t>& cov,
              const Eigen::Ref<const mat_t>& neg_precQ,
              const Eigen::Ref<const vec_t>& logprecQdet,
              const Eigen::Ref<const vec_t>& logprior, value_t opt_tol,
              value_t logit_p1)
        : arm_base_t(ARMS, n_arm_samples),
          base_t(),
          n_arm_samples(n_arm_samples),
          efficacy_thresholds(efficacy_thresholds),
          quad_wts(quad_wts),
          cov(cov),
          neg_precQ(neg_precQ),
          logprecQdet(logprecQdet),
          logprior(logprior),
          opt_tol2(opt_tol * opt_tol),
          logit_p1(logit_p1) {
        assert(efficacy_thresholds.size() == ARMS);
        set_critical_values(cv);
    }

    using base_t::critical_values;
    void set_critical_values(const Eigen::Ref<const colvec_type<value_t>>& cv) {
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
    auto make_kevlar_bound_state(const _GridRangeType& gr) const {
        return kevlar_bound_state_t<_GridRangeType>(n_arm_samples, gr);
    }

    vec_t get_posterior_exceedance_probs(const vec_t& y) const {
        assert((y.array() >= 0).all());
        assert((y.array() <= n_arm_samples).all());
        size_t nsig2 = quad_wts.size();

        vec_t exceedances(ARMS);
        exceedances.setZero();

        double sigma2_integral = 0.0;
        for (int s = 0; s < nsig2; s++) {
            std::array<double, ARMS> t{};
            std::array<double, ARMS> grad;
            std::array<double, ARMS> hess_diag;
            std::array<double, ARMS> step;
            std::array<double, ARMS> theta_sigma;

            std::array<std::array<double, ARMS>, ARMS> tmp;
            std::array<std::array<double, ARMS>, ARMS> hess_inv;
            bool converged = false;
            int opt_iter = 0;

            // Newton's method for finding the mode of the joint distribution.
            for (; opt_iter < 20; opt_iter++) {
                // construct gradient and hessian.
                // grad = y - n * (e^{t + L}) / (e^{t + L} + 1)
                //        -Q(t - mu_0)
                //   where L = logit_p1
                //
                // The hessian we have can be written as the sum of a full rank
                // matrix and a diagonal term. Importantly, we already know the
                // inverse of the full rank matrix.
                //
                // To invert this type of matrix, we  will use an iterative
                // application of the Sherman-Morrison formula to compute the
                // inverse:
                // (A + uv)^{-1} = A^{-1} - \frac{A^{-1} u v^T A^{-1}}{1 + v^T
                // A^{-1} u}
                //
                // A^{-1} = cov
                // u = (n e^{t + L} / (e^{t + L} + 1)^2) e_i
                // v = e_i
                //   where
                //     e_i = vector of all zeros except entry i = 1
                //     L = logit_p1
                for (int i = 0; i < ARMS; i++) {
                    auto theta_adj = t[i] + logit_p1;
                    auto exp_theta_adj = std::exp(theta_adj);
                    auto C = 1.0 / (exp_theta_adj + 1);
                    // TODO: n_arm_samples is going to need to change through
                    // the multiple interim analyses?
                    auto nCeta = n_arm_samples * C * exp_theta_adj;
                    grad[i] = y[i] - nCeta;
                    for (int j = 0; j < ARMS; j++) {
                        // int idx = s * ARMS * ARMS + i * ARMS + j;
                        int idx = (s * ARMS + i) * ARMS + j;
                        grad[i] += neg_precQ.data()[(s * ARMS + i) * ARMS + j] *
                                   (t[j] - mu_0);
                    }

                    for (int j = 0; j < ARMS; j++) {
                        hess_inv[i][j] = cov.data()[(s * ARMS + i) * ARMS + j];
                    }
                    hess_diag[i] = nCeta * C;
                }

                // invert hessian by repeatedly using the Sherman-Morrison
                // formula:
                for (int k = 0; k < ARMS; k++) {
                    double offset =
                        hess_diag[k] / (1 + hess_diag[k] * hess_inv[k][k]);
                    for (int i = 0; i < ARMS; i++) {
                        for (int j = 0; j < ARMS; j++) {
                            tmp[i][j] =
                                offset * hess_inv[k][i] * hess_inv[j][k];
                        }
                    }
                    for (int i = 0; i < ARMS; i++) {
                        for (int j = 0; j < ARMS; j++) {
                            hess_inv[i][j] -= tmp[i][j];
                        }
                    }
                }

                // take newton step
                double step_len2 = 0.0;
                for (int i = 0; i < ARMS; i++) {
                    step[i] = 0.0;
                    for (int j = 0; j < ARMS; j++) {
                        step[i] += hess_inv[i][j] * grad[j];
                    }
                    step_len2 += step[i] * step[i];
                    t[i] += step[i];
                }

                // check for convergence.
                if (step_len2 < opt_tol2) {
                    converged = true;
                    break;
                }
            }
            assert(converged);

            // calculate log joint distribution
            // joint = (t+logit_p1)y - n(log(e^(t+logit_p1) + 1))
            //         -(t-mu)^{T}Q(t-mu) + 0.5 * log(det(Q)) + prior
            double logjoint = logprecQdet[s] + logprior[s];
            std::array<double, ARMS> tmm0;
            for (int i = 0; i < ARMS; i++) {
                tmm0[i] = t[i] - mu_0;
            }
            for (int i = 0; i < ARMS; i++) {
                double quadsum = 0.0;
                for (int j = 0; j < ARMS; j++) {
                    quadsum +=
                        neg_precQ.data()[(s * ARMS + i) * ARMS + j] * tmm0[j];
                }
                logjoint += 0.5 * tmm0[i] * quadsum;

                auto theta_adj = t[i] + logit_p1;
                logjoint += theta_adj * y[i] -
                            n_arm_samples * std::log(std::exp(theta_adj) + 1);
            }

            // marginal std dev is the sqrt of the hessian diagonal.
            // we need to store this because we're going to destroy hess_inv in
            // the determinant calculation
            for (int i = 0; i < ARMS; i++) {
                theta_sigma[i] = std::sqrt(hess_inv[i][i]);
            }

            // determinant of hessian (this destroys hess_inv so don't use
            // hess_inv after here!)
            double c;
            double det = 1;
            for (int i = 0; i < ARMS; i++) {
                for (int k = i + 1; k < ARMS; k++) {
                    c = hess_inv[k][i] / hess_inv[i][i];
                    for (int j = i; j < ARMS; j++) {
                        hess_inv[k][j] = hess_inv[k][j] - c * hess_inv[i][j];
                    }
                }
            }
            for (int i = 0; i < ARMS; i++) {
                det *= hess_inv[i][i];
            }

            // calculate p(sigma^2 | y) along with its normalization
            auto sigma2_post = std::exp(logjoint + 0.5 * std::log(det));
            sigma2_integral += sigma2_post * quad_wts[s];

            // calculate exceedance probabilities, integrated over sigma2
            for (int i = 0; i < ARMS; i++) {
                double normalized =
                    (efficacy_thresholds[i] - t[i]) / theta_sigma[i];
                double exc_sigma2 = 0.5 * (erf(-normalized * M_SQRT1_2) + 1);
                exceedances[i] += exc_sigma2 * sigma2_post * quad_wts[s];
            }
        }

        // normalize the sigma2_post distribution
        double inv_sigma2_integral = 1.0 / sigma2_integral;
        for (int i = 0; i < ARMS; i++) {
            exceedances[i] *= inv_sigma2_integral;
        }
        return exceedances;
    }
};

template <class ValueType, int ARMS>
template <class _GenType, class _ValueType, class _UIntType,
          class _GridRangeType>
struct BerryINLA<ValueType, ARMS>::SimGlobalState
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
    using model_t = BerryINLA;
    const model_t& model;

   public:
    SimGlobalState(const model_t& model, const grid_range_t& grid_range)
        : base_t(model.n_arm_samples, grid_range), model(model) {}

    std::unique_ptr<typename interface_t::sim_state_t> make_sim_state(
        size_t seed) const override {
        return std::make_unique<sim_state_t>(*this, seed);
    }
};

template <class ValueType, int ARMS>
template <class _GenType, class _ValueType, class _UIntType,
          class _GridRangeType>
struct BerryINLA<ValueType, ARMS>::SimGlobalState<
    _GenType, _ValueType, _UIntType, _GridRangeType>::SimState
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
        const auto& gr = outer_.grid_range();

        assert(gr.n_params() == ARMS);

        const auto n_arm_samples = outer_.model.n_arm_samples;
        const auto& critical_values = outer_.model.critical_values();

        size_t pos = 0;
        for (size_t grid_i = 0; grid_i < gr.n_gridpts(); ++grid_i) {
            auto bits_i = bits.col(grid_i);

            vec_t y(ARMS);
            for (int i = 0; i < y.size(); ++i) {
                const auto& ss_i = base_t::sufficient_stats_arm(i);
                y(i) = static_cast<value_t>(ss_i(bits_i[i]));
            }

            vec_t exceedance = outer_.model.get_posterior_exceedance_probs(y);

            for (size_t n_t = 0; n_t < gr.n_tiles(grid_i); ++n_t, ++pos) {
                value_t max_null_prob_exceed = 0;
                for (int arm_i = 0; arm_i < ARMS; ++arm_i) {
                    if (gr.check_null(pos, arm_i)) {
                        max_null_prob_exceed =
                            std::max(max_null_prob_exceed, exceedance[arm_i]);
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
}  // namespace kevlar
