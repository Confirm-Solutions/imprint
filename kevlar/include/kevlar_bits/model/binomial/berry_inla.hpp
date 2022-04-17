#pragma once
#include <algorithm>
#include <Eigen/Dense>
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

struct BerryINLA : SimpleSelection<double> {
    using base_t = SimpleSelection<double>;
    using typename base_t::value_t;

   private:
    using vec_t = colvec_type<value_t>;
    using mat_t = mat_type<value_t>;

    static constexpr value_t alpha_prior = 0.0005;
    static constexpr value_t beta_prior = 0.000005;
    static constexpr double mu_sig_sq = 100;

    // variables dependent solely on the quadrature points (independent of the
    // data).
    vec_t quad_pts;
    vec_t quad_wts;
    vec_t log_prior;
    mat_t cov;
    mat_t neg_precQ;
    vec_t logprecQdet;
    vec_t logprior;
    value_t opt_tol;

   public:
    const vec_t efficacy_thresholds;
    const int n_arms;
    const int n_arm_samples;

    template <class _GenType, class _UIntType, class _GridRangeType>
    struct SimGlobalState;

    template <class _GenType, class _UIntType, class _GridRangeType>
    using sim_global_state_t =
        SimGlobalState<_GenType, _UIntType, _GridRangeType>;

    template <class _GridRangeType>
    using kevlar_bound_state_t = KevlarBoundStateFixedNDefault<_GridRangeType>;

    BerryINLA(int n_arms, int n_arm_samples, const Eigen::Ref<const vec_t>& cv,
              const Eigen::Ref<const vec_t>& efficacy_thresholds,
              const Eigen::Ref<const vec_t>& quad_pts,
              const Eigen::Ref<const vec_t>& quad_wts,
              const Eigen::Ref<const mat_t>& cov,
              const Eigen::Ref<const mat_t>& neg_precQ,
              const Eigen::Ref<const vec_t>& logprecQdet,
              const Eigen::Ref<const vec_t>& logprior, value_t opt_tol)
        : SimpleSelection<double>(n_arms, n_arm_samples, 0, cv),
          n_arms(n_arms),
          n_arm_samples(n_arm_samples),
          efficacy_thresholds(efficacy_thresholds),
          quad_pts(quad_pts),
          quad_wts(quad_wts),
          cov(cov),
          neg_precQ(neg_precQ),
          logprecQdet(logprecQdet),
          logprior(logprior),
          opt_tol(opt_tol) {
        assert(efficacy_thresholds.size() == n_arms);
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
        return sim_global_state_t<_GenType, _UIntType, _GridRangeType>(
            *this, grid_range);
    }

    template <class _GridRangeType>
    auto make_kevlar_bound_state(const _GridRangeType& gr) const {
        return kevlar_bound_state_t<_GridRangeType>(n_arm_samples, gr);
    }

    vec_t get_posterior_exceedance_probs(const vec_t& phat) const {
        assert((phat.array() >= 0).all());
        assert((phat.array() <= 1).all());
        const vec_t thetahat = logit(phat.array());
        ASSERT_GOOD(thetahat);
    }
};

template <class _GenType, class _UIntType, class _GridRangeType>
struct BerryINLA::SimGlobalState
    : SimGlobalStateFixedNDefault<_GenType, double, _UIntType, _GridRangeType> {
    struct SimState;

    using base_t = SimGlobalStateFixedNDefault<_GenType, double, _UIntType,
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
    SimGlobalState(const BerryINLA& model, const grid_range_t& grid_range)
        : base_t(model.n_arm_samples, grid_range), model(model) {}

    std::unique_ptr<typename interface_t::sim_state_t> make_sim_state()
        const override {
        return std::make_unique<sim_state_t>(*this);
    }
};

template <class _GenType, class _UIntType, class _GridRangeType>
struct BerryINLA::SimGlobalState<_GenType, _UIntType, _GridRangeType>::SimState
    : base_t::sim_state_t {
   private:
    using outer_t = SimGlobalState;

   public:
    using base_t = typename outer_t::base_t::sim_state_t;
    using typename base_t::interface_t;

   private:
    const outer_t& outer_;

   public:
    SimState(const outer_t& sgs) : base_t(sgs), outer_(sgs) {}

    void simulate(gen_t& gen,
                  Eigen::Ref<colvec_type<uint_t>> rej_len) override {
        std::cout << "lol" << std::endl;
        base_t::generate_data(gen);
        base_t::generate_sufficient_stats();
        std::cout << "gen" << std::endl;
        return;
        const auto& bits = outer_.bits();
        const auto& gr_view = outer_.grid_range();
        const auto n_params = gr_view.n_params();  // same as n_arms
        const auto& model = outer_.model;
        const auto n_arm_samples = model.n_arm_samples;
        const auto& critical_values = model.critical_values();
        std::cout << model.cov.rows() << " " << model.cov.cols() << std::endl;

        size_t pos = 0;
        for (size_t grid_i = 0; grid_i < gr_view.n_gridpts(); ++grid_i) {
            auto bits_i = bits.col(grid_i);

            vec_t phat(n_params);
            for (int i = 0; i < phat.size(); ++i) {
                const auto& ss_i = base_t::sufficient_stats_arm(i);
                phat(i) = static_cast<value_t>(ss_i(bits_i[i])) / n_arm_samples;
            }
            return;

            vec_t posterior_exceedance_probs =
                outer_.model.get_posterior_exceedance_probs(phat);
            return;

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
}  // namespace kevlar