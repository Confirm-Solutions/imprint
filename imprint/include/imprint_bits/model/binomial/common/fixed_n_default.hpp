#pragma once
#include <imprint_bits/distribution/binomial.hpp>
#include <imprint_bits/distribution/uniform.hpp>
#include <imprint_bits/model/base.hpp>
#include <imprint_bits/util/algorithm.hpp>
#include <imprint_bits/util/macros.hpp>
#include <imprint_bits/util/types.hpp>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

namespace imprint {
namespace model {
namespace binomial {

/*
 * This class represents the default cache for all binomial models.
 * By definition, binomial models are those that assume the data
 * is drawn from a binomial distribution independently across arms.
 * This class further assumes that all binomials have common size n.
 * The default eta transformation is the identity function,
 * so it assumes grid-points lie in the natural parameter space.
 */
template <class GenType, class ValueType, class UIntType, class GridRangeType>
struct SimGlobalStateFixedNDefault : SimGlobalStateBase<ValueType, UIntType> {
    struct SimState;

    using base_t = SimGlobalStateBase<ValueType, UIntType>;
    using typename base_t::interface_t;
    using typename base_t::uint_t;
    using typename base_t::value_t;
    using gen_t = GenType;
    using grid_range_t = GridRangeType;

    using sim_state_t = SimState;

   private:
    using binom_t = distribution::Binomial<int>;
    using vec_t = colvec_type<value_t>;
    using uvec_t = colvec_type<uint_t>;
    using mat_t = mat_type<value_t>;
    using umat_t = mat_type<uint_t>;

    const size_t n_arm_samples_;       // number of samples for each arm
    const grid_range_t& grid_range_;   // save reference to original grid range
    std::vector<vec_t> probs_unique_;  // probs_unique_[i] = unique prob vector
                                       // sorted (ascending) for arm i.
    uvec_t strides_;  // strides_[i] = number of unique probs for arm i-1 with 0
                      // for arm -1.
    umat_t gbits_;    // jth grid-point's ith coordinate is given by
                      // probs_unique_[i][gbits_(i,j)]
    size_t n_total_uniques_ = 0;  // total number of unique probability values

    IMPRINT_STRONG_INLINE
    auto n_params() const { return grid_range_.n_params(); }

    /*
     * Populates the private members.
     */
    void construct() {
        const auto n_params = grid_range_.n_params();

        // resize all other internal quantities
        probs_unique_.resize(n_params);
        strides_.resize(n_params + 1);
        gbits_.resize(n_params, grid_range_.n_gridpts());

        const auto& thetas = grid_range_.thetas();

        // populate set of unique theta values for each arm
        std::unordered_map<value_t, uint_t> pu_to_idx;
        std::set<value_t> prob_set;
        colvec_type<value_t> prob;
        auto& bits = gbits_;

        strides_[0] = 0;  // initialize stride to arm 0.

        for (size_t i = 0; i < n_params; ++i) {
            pu_to_idx.clear();
            prob_set.clear();

            // insert all prob values in arm i into the set
            prob = binom_t::natural_to_mean(thetas.row(i).array());
            for (int j = 0; j < prob.size(); ++j) {
                prob_set.insert(prob[j]);
            }

            // create a mapping from unique prob values to order idx
            {
                int j = 0;
                for (auto p : prob_set) {
                    pu_to_idx[p] = j++;
                }
            }

            // increment number of total uniques
            n_total_uniques_ += prob_set.size();

            // copy number of uniques
            strides_[i + 1] = strides_[i] + prob_set.size();

            // copy unique prob values into vector
            probs_unique_[i].resize(prob_set.size());
            std::copy(prob_set.begin(), prob_set.end(),
                      probs_unique_[i].data());

            // populate bits for current arm
            auto bits_i = bits.row(i);
            for (int j = 0; j < bits_i.size(); ++j) {
                bits_i(j) = pu_to_idx[prob(j)];
            }
        }
    }

   public:
    SimGlobalStateFixedNDefault(size_t n_arm_samples,
                                const grid_range_t& grid_range)
        : n_arm_samples_(n_arm_samples), grid_range_(grid_range) {
        construct();
    }

    IMPRINT_STRONG_INLINE
    const auto& bits() const { return gbits_; }

    IMPRINT_STRONG_INLINE
    const auto& grid_range() const { return grid_range_; }

    IMPRINT_STRONG_INLINE
    auto stride(size_t i) const { return strides_[i]; }

    IMPRINT_STRONG_INLINE
    const auto& probs_unique_arm(size_t i) const { return probs_unique_[i]; }
};

/*
 * This class is the corresponding simulation state
 * for the fixed-n default case.
 * Assuming everything in the global state,
 * this class assumes some default behavior of
 *  - generating data given the whole grid-range
 *  - computing sufficient statistics
 *  - computing score
 */
template <class GenType, class ValueType, class UIntType, class GridRangeType>
struct SimGlobalStateFixedNDefault<GenType, ValueType, UIntType,
                                   GridRangeType>::SimState
    : SimGlobalStateFixedNDefault::base_t::sim_state_t {
   private:
    using outer_t = SimGlobalStateFixedNDefault;

   public:
    using base_t = typename outer_t::base_t::sim_state_t;
    using typename base_t::interface_t;

   private:
    using uniform_t = distribution::Uniform<value_t>;

    const outer_t& outer_;
    uniform_t uniform_;
    mat_type<value_t> uniform_randoms_;  // uniform rng
    colvec_type<uint_t>
        sufficient_stat_;  // sufficient statistic table for each
                           // arm and prob value sufficient_stat_(i,j) =
                           // suff stat at unique prob i at arm j.
    gen_t gen_;

    template <bool do_const>
    IMPRINT_STRONG_INLINE auto sufficient_stats_arm(size_t j) const {
        using vec_t = std::conditional_t<do_const, const colvec_type<uint_t>,
                                         colvec_type<uint_t>>;
        auto& ss_casted = const_cast<vec_t&>(sufficient_stat_);
        return Eigen::Map<vec_t>(ss_casted.data() + outer_.strides_[j],
                                 outer_.strides_[j + 1] - outer_.strides_[j]);
    }

   public:
    SimState(const outer_t& outer, size_t seed)
        : outer_(outer), uniform_(0., 1.), gen_(seed) {}

    /*
     * Returns a reference to the RNG.
     */
    auto& rng() { return gen_; }

    /*
     * Returns a read-only reference to the uniform randoms.
     * Note that if generate_sufficient_stats has been called before,
     * each column will be sorted uniform randoms.
     */
    IMPRINT_STRONG_INLINE
    auto& uniform_randoms() { return uniform_randoms_; }

    IMPRINT_STRONG_INLINE
    const auto& uniform_randoms() const { return uniform_randoms_; }

    /*
     * Creates a view of jth arm sufficient stats counts.
     * Note that 0 <= j < n_arms.
     */
    IMPRINT_STRONG_INLINE
    auto sufficient_stats_arm(size_t j) const {
        return sufficient_stats_arm<true>(j);
    }

    /*
     * Generate uniform random variables of shape (n_arm_samples, n_params).
     */
    IMPRINT_STRONG_INLINE
    void generate_data() {
        const auto n_arm_samples = outer_.n_arm_samples_;
        const auto n_params = outer_.n_params();
        uniform_.sample(n_arm_samples, n_params, gen_, uniform_randoms_);
    }

    /*
     * Generates sufficient statistic for each arm
     * and for each unique probability.
     */
    IMPRINT_STRONG_INLINE
    void generate_sufficient_stats() {
        const auto n_params = outer_.n_params();
        const auto n_total_uniques = outer_.n_total_uniques_;

        // sort each column of the uniforms
        sort_cols(uniform_randoms_);

        sufficient_stat_.resize(n_total_uniques);

        // output cumulative count of uniforms < p
        // for each unique probability value p.
        for (size_t i = 0; i < n_params; ++i) {
            auto ss_i = sufficient_stats_arm<false>(i);

            accum_count(uniform_randoms_.col(i), outer_.probs_unique_[i], ss_i);
        }
    }

    /*
     * Computes the score of a binomial distribution at gridpt_idx.
     */
    void score(size_t gridpt_idx,
               Eigen::Ref<colvec_type<value_t>> out) const override {
        assert(out.size() == outer_.n_params());
        for (int k = 0; k < out.size(); ++k) {
            auto ss_a = sufficient_stats_arm<false>(k);
            auto unique_idx = outer_.gbits_(k, gridpt_idx);
            out[k] = binom_t::score(ss_a(unique_idx), outer_.n_arm_samples_,
                                    outer_.probs_unique_arm(k)[unique_idx]);
        }
    }
};

/*
 * This class represents the default imprint bound state for all binomial
 * models. See the assumptions of binomial model in global state class above.
 */
template <class GridRangeType>
struct ImprintBoundStateFixedNDefault
    : ImprintBoundStateBase<typename GridRangeType::value_t> {
    using grid_range_t = GridRangeType;
    using base_t = ImprintBoundStateBase<typename grid_range_t::value_t>;
    using typename base_t::interface_t;
    using typename base_t::value_t;

   private:
    using binom_t = distribution::Binomial<int>;

    const grid_range_t& grid_range_;
    size_t n_arm_samples_;
    colvec_type<value_t> p_buffer_;

    template <bool do_const>
    auto p_slice(size_t slice) const {
        using mat_t = std::conditional_t<do_const, const mat_type<value_t>,
                                         mat_type<value_t>>;
        using vec_t = std::conditional_t<do_const, const colvec_type<value_t>,
                                         colvec_type<value_t>>;
        auto& p_buffer_cast = const_cast<vec_t&>(p_buffer_);
        const auto mat_size = grid_range_.n_params() * grid_range_.n_gridpts();
        return Eigen::Map<mat_t>(p_buffer_cast.data() + mat_size * slice,
                                 grid_range_.n_params(),
                                 grid_range_.n_gridpts());
    }

    auto p_lower() const { return p_slice<true>(0); }
    auto p() const { return p_slice<true>(1); }
    auto p_upper() const { return p_slice<true>(2); }

   public:
    ImprintBoundStateFixedNDefault(size_t n_arm_samples,
                                   const grid_range_t& grid_range)
        : grid_range_(grid_range),
          n_arm_samples_(n_arm_samples),
          p_buffer_(grid_range.n_params() * grid_range.n_gridpts() * 3) {
        const auto& thetas = grid_range.thetas();
        const auto& radii = grid_range.radii();
        p_slice<false>(0) =
            binom_t::natural_to_mean(thetas.array() - radii.array());
        p_slice<false>(1) = binom_t::natural_to_mean(thetas.array());
        p_slice<false>(2) =
            binom_t::natural_to_mean(thetas.array() + radii.array());
    }

    /*
     * Note that grid-point information is not used.
     */
    void apply_eta_jacobian(size_t,
                            const Eigen::Ref<const colvec_type<value_t>>& v,
                            Eigen::Ref<colvec_type<value_t>> out) override {
        assert(v.size() == n_natural_params());
        assert(v.size() == out.size());
        out = v;
    }

    value_t covar_quadform(
        size_t gridpt_idx,
        const Eigen::Ref<const colvec_type<value_t>>& v) override {
        assert(v.size() == n_natural_params());
        return binom_t::covar_quadform(n_arm_samples_,
                                       p().col(gridpt_idx).array(), v.array());
    }

    /*
     * Note that tile information is not used in this bound.
     */
    value_t hessian_quadform_bound(
        size_t gridpt_idx, size_t,
        const Eigen::Ref<const colvec_type<value_t>>& v) override {
        assert(v.size() == n_natural_params());

        auto p_lower_ = p_lower().col(gridpt_idx);
        auto p_upper_ = p_upper().col(gridpt_idx);

        value_t hess_bd = 0;
        for (int k = 0; k < v.size(); ++k) {
            auto v_sq = v[k] * v[k];
            if (p_lower_[k] <= 0.5 && 0.5 <= p_upper_[k]) {
                hess_bd += 0.25 * v_sq;
            } else {
                auto lower = p_lower_[k] - 0.5;  // shift away center
                auto upper = p_upper_[k] - 0.5;  // shift away center
                // max of p(1-p) occurs for whichever p is closest to 0.5.
                bool max_at_upper = (std::abs(upper) < std::abs(lower));
                auto max_endpt = max_at_upper ? p_upper_[k] : p_lower_[k];
                hess_bd += max_endpt * (1. - max_endpt) * v_sq;
            }
        }
        return hess_bd * n_arm_samples_;
    }

    size_t n_natural_params() const override { return grid_range_.n_params(); }
};

}  // namespace binomial
}  // namespace model
}  // namespace imprint
