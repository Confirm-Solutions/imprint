#pragma once
#include <algorithm>
#include <boost/math/special_functions/beta.hpp>
#include <imprint_bits/distribution/binomial.hpp>
#include <imprint_bits/model/base.hpp>
#include <imprint_bits/model/binomial/common/fixed_n_default.hpp>
#include <imprint_bits/model/fixed_single_arm_size.hpp>
#include <imprint_bits/util/macros.hpp>

namespace imprint {
namespace model {
namespace binomial {

template <class ValueType>
struct Thompson : FixedSingleArmSize, ModelBase<ValueType> {
    using arm_base_t = FixedSingleArmSize;
    using base_t = ModelBase<ValueType>;
    using typename base_t::value_t;

   private:
    const value_t alpha_prior_;
    const value_t beta_prior_;
    const value_t p_thresh_;

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

    /*
     * @param   n_arm_samples   max number of patients in each arm.
     */
    Thompson(size_t n_arm_samples, value_t alpha_prior, value_t beta_prior,
             value_t p_thresh, const Eigen::Ref<const colvec_type<value_t>>& cv)
        : arm_base_t(2, n_arm_samples),
          base_t(),
          alpha_prior_(alpha_prior),
          beta_prior_(beta_prior),
          p_thresh_(p_thresh) {
        critical_values(cv);
    }

    using arm_base_t::n_arm_samples;
    using arm_base_t::n_arms;

    using base_t::critical_values;
    void critical_values(const Eigen::Ref<const colvec_type<value_t>>& cv) {
        auto& cv_ = base_t::critical_values();
        cv_ = cv;
        std::sort(cv_.data(), cv_.data() + cv_.size(), std::greater<value_t>());
    }

    value_t alpha_prior() const { return alpha_prior_; }
    value_t beta_prior() const { return beta_prior_; }
    value_t p_threshold() const { return p_thresh_; }

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
struct Thompson<ValueType>::SimGlobalState
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
    using binom_t = distribution::Binomial<uint_t>;
    using model_t = Thompson;
    const model_t& model_;

    const model_t& model() const { return model_; }

   public:
    SimGlobalState(const model_t& model, const grid_range_t& grid_range)
        : base_t(model.n_arm_samples(), grid_range), model_(model) {}

    std::unique_ptr<typename interface_t::sim_state_t> make_sim_state(
        size_t seed) const override {
        return std::make_unique<sim_state_t>(*this, seed);
    }
};

template <class ValueType>
template <class _GenType, class _ValueType, class _UIntType,
          class _GridRangeType>
struct Thompson<ValueType>::SimGlobalState<_GenType, _ValueType, _UIntType,
                                           _GridRangeType>::SimState
    : base_t::sim_state_t {
   private:
    using outer_t = SimGlobalState;

   public:
    using base_t = typename outer_t::base_t::sim_state_t;
    using typename base_t::interface_t;

   private:
    using unif_t = distribution::Uniform<value_t>;
    const outer_t& outer_;
    colvec_type<value_t> g_sums_;  // g_sums_[i] = sum of i iid Gamma(1,1).
    std::gamma_distribution<value_t> gamma_a_;  // Gamma(a,1)
    std::gamma_distribution<value_t> gamma_b_;  // Gamma(b,1)
    std::gamma_distribution<value_t> gamma_1_;  // Gamma(1,1)
    value_t end_left_ = 0;                      // end left gamma posterior
    value_t end_right_ = 0;                     // end right gamma posterior

    IMPRINT_STRONG_INLINE
    auto compute_posterior(uint_t n, uint_t s) {
        return (g_sums_(s) + end_left_) / (g_sums_(n) + end_left_ + end_right_);
    }

    IMPRINT_STRONG_INLINE
    void generate_data() {
        // generate uniforms
        base_t::generate_data();

        // get rng
        auto& gen = base_t::rng();

        // cache gamma values
        end_left_ = gamma_a_(gen);
        end_right_ = gamma_b_(gen);
        g_sums_(0) = 0;
        for (int i = 1; i < g_sums_.size(); ++i) {
            g_sums_(i) = g_sums_(i - 1) + gamma_1_(gen);
        }
    }

    template <class BitsType>
    void internal(const BitsType& bits,
                  colvec_type<value_t, 2>& posterior_exceedance_probs) {
        // compute alpha, beta posterior parameters
        colvec_type<uint_t, 2> n_action_arms;
        colvec_type<uint_t, 2> successes;
        colvec_type<value_t, 2> posterior;

        n_action_arms.setZero();
        successes.setZero();

        const auto& unifs = base_t::uniform_randoms();

        // iterate through each patient
        auto max_iter = outer_.model().n_arm_samples();
        for (uint_t i = 0; i < max_iter; ++i) {
            for (uint_t j = 0; j < 2; ++j) {
                posterior(j) =
                    compute_posterior(n_action_arms(j), successes(j));
            }

            bool action = (posterior(1) > posterior(0));

            for (uint_t j = 0; j < 2; ++j) {
                bool action_is_j = (action == j);
                successes(j) +=
                    (action_is_j &&
                     (outer_.probs_unique_arm(j)(bits[j]) > unifs(i, j)));
                n_action_arms(j) += action_is_j;
            }
        }

        colvec_type<value_t, 2> alpha_posterior;
        colvec_type<value_t, 2> beta_posterior;
        alpha_posterior.array() = successes.template cast<value_t>().array() +
                                  outer_.model().alpha_prior();
        beta_posterior.array() =
            (n_action_arms - successes).template cast<value_t>().array() +
            outer_.model().beta_prior();

        // compute posterior exceedance probs
        auto p_thresh = outer_.model().p_threshold();
        for (uint_t i = 0; i < posterior_exceedance_probs.size(); ++i) {
            posterior_exceedance_probs[i] = boost::math::ibetac(
                alpha_posterior[i], beta_posterior[i], p_thresh);
        }
    }

   public:
    SimState(const outer_t& sgs, size_t seed)
        : base_t(sgs, seed),
          outer_(sgs),
          g_sums_(outer_.model().n_arm_samples() + 1),
          gamma_a_(sgs.model().alpha_prior()),
          gamma_b_(sgs.model().beta_prior()),
          gamma_1_() {}

    void simulate(Eigen::Ref<colvec_type<uint_t>> rej_len) override {
        // generate all possible gamma outcomes and uniforms
        generate_data();

        const auto& sgs = outer_;
        const auto& bits = sgs.bits();
        const auto& gr_view = sgs.grid_range();

        size_t pos = 0;
        for (int i = 0; i < gr_view.n_gridpts(); ++i) {
            const auto bits_i = bits.col(i);

            colvec_type<value_t, 2> posterior_exceedance_probs;
            internal(bits_i, posterior_exceedance_probs);

            // get max posterior exceedance prob among all arms
            Eigen::Index max_arm;
            value_t max_pep = posterior_exceedance_probs.maxCoeff(&max_arm);

            const auto n_ts = gr_view.n_tiles(i);
            for (size_t n_t = 0; n_t < n_ts; ++n_t, ++pos) {
                // if selected arm is not null
                if (!gr_view.check_null(pos, max_arm)) {
                    rej_len[pos] = 0;
                    continue;
                }

                // find first time when max pep is > critical value
                const auto& cvs = outer_.model().critical_values();
                size_t j = 0;
                for (; j < cvs.size(); ++j) {
                    if (max_pep > cvs[j]) break;
                }

                rej_len[pos] = cvs.size() - j;
            }
        }

        // generate sufficient stats
        // this is needed for score function to work properly.
        // Must come after previous loop since this function
        // internally sorts the uniforms.
        base_t::generate_sufficient_stats();

        assert(rej_len.size() == pos);
    }

    using base_t::score;
};

}  // namespace binomial
}  // namespace model
}  // namespace imprint
