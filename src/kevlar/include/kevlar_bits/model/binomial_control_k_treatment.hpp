#pragma once
#include <kevlar_bits/util/types.hpp>
#include <kevlar_bits/util/algorithm.hpp>
#include <kevlar_bits/util/math.hpp>
#include <kevlar_bits/util/macros.hpp>
#include <kevlar_bits/model/base.hpp>
#include <limits>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <random>
#include <unordered_map>
#include <vector>

namespace kevlar {

/* Forward declaration */
template <class ValueType, class UIntType, class GridRangeType>
struct BinomialControlkTreatment;

namespace internal {

template <class T>
struct traits;

template <class ValueType, class UIntType, class GridRangeType>
struct traits<BinomialControlkTreatment<ValueType, UIntType, GridRangeType>> {
    using value_t = ValueType;
    using uint_t = UIntType;
    using grid_range_t = GridRangeType;
    using state_t =
        typename BinomialControlkTreatment<ValueType, UIntType,
                                           GridRangeType>::StateType;
};

}  // namespace internal

template <class ValueType, class UIntType, class GridRangeType>
struct BinomialControlkTreatment
    : ControlkTreatmentBase,
      ModelBase<ValueType, UIntType, GridRangeType> {
   private:
    using static_interface_t = ControlkTreatmentBase;

   public:
    using value_t = ValueType;
    using uint_t = UIntType;
    using grid_range_t = GridRangeType;
    using base_t = static_interface_t;
    using model_base_t = ModelBase<ValueType, UIntType, GridRangeType>;

    struct StateType : ModelStateBase<value_t, uint_t, grid_range_t> {
       private:
        using outer_t = BinomialControlkTreatment;
        const outer_t& outer_;

       public:
        using model_state_base_t =
            ModelStateBase<value_t, uint_t, grid_range_t>;

        StateType(const outer_t& outer)
            : model_state_base_t(outer), outer_(outer), unif_dist_(0., 1.) {
            for (auto& pu : outer_.probs_unique_) {
                n_total_uniques_ += pu.size();
            }
        }

        /*
         * Generate RNG for the simulation.
         * Generate U(0,1) for each patient in each arm.
         */
        template <class GenType>
        void gen_rng(GenType&& gen) {
            static_interface_t::uniform(outer_.n_samples(), outer_.n_arms(),
                                        gen, unif_dist_, unif_);
        }

        /*
         * Overrides the necessary RNG generator.
         * TODO: generalize mt19937 somehow.
         */
        void gen_rng(std::mt19937& gen) override {
            gen_rng<std::mt19937&>(gen);
        }

        /*
         * Generates sufficient statistic for each arm under all possible grid
         * points.
         */
        void gen_suff_stat() override {
            size_t k = outer_.n_arms() - 1;
            size_t n = outer_.n_samples();
            size_t ph2_size = outer_.ph2_size_;
            size_t ph3_size = n - ph2_size;

            // grab the block of uniforms associated with Phase II/III for
            // treatments.
            auto ph2_unif = unif_.block(0, 1, ph2_size, k);
            auto ph3_unif = unif_.block(ph2_size, 1, ph3_size, k);

            // grab control uniforms
            auto control_unif = unif_.col(0);

            // sort each column of each block.
            sort_cols(ph2_unif);
            sort_cols(ph3_unif);
            sort_cols(control_unif);

            suff_stat_.resize(n_total_uniques_);
            Eigen::Map<colvec_type<uint_t>> control_counts(
                suff_stat_.data(), outer_.probs_unique_[0].size());
            Eigen::Map<colvec_type<uint_t>> ph3_counts(
                suff_stat_.data() + control_counts.size(),
                suff_stat_.size() - control_counts.size());
            ph2_counts_.resize(ph3_counts.size());

            // output cumulative count of uniforms < outer_.prob_[k] into counts
            // object.
            cum_count(control_unif, outer_.probs_unique_[0], control_counts);
            {
                size_t n_skip = 0;
                for (size_t i = 0; i < k; ++i) {
                    Eigen::Map<colvec_type<uint_t>> ph2_counts_i(
                        ph2_counts_.data() + n_skip,
                        outer_.probs_unique_[i + 1].size());
                    Eigen::Map<colvec_type<uint_t>> ph3_counts_i(
                        ph3_counts.data() + n_skip,
                        outer_.probs_unique_[i + 1].size());
                    cum_count(ph2_unif.col(i), outer_.probs_unique_[i + 1],
                              ph2_counts_i);
                    cum_count(ph3_unif.col(i), outer_.probs_unique_[i + 1],
                              ph3_counts_i);

                    n_skip += ph2_counts_i.size();
                }
            }

            // final update on suff_stat
            suff_stat_.tail(suff_stat_.size() - control_counts.size()) +=
                ph2_counts_;
        }

        /*
         * @param   rej_len      vector of integers to store number of models
         * that reject.
         */
        void rej_len(Eigen::Ref<colvec_type<uint_t>> rej_len) override {
            auto& bits = outer_.gbits_;
            auto& gr_view = outer_.grid_range();

            int pos = 0;
            for (int i = 0; i < outer_.n_gridpts(); ++i) {
                auto bits_i = bits.col(i);

                // Phase II
                int a_star =
                    -1;  // selected arm with highest Phase II response count.
                int max_count = -1;  // maximum Phase II response count.
                for (int j = 1; j < bits_i.size(); ++j) {
                    int prev_count = max_count;
                    Eigen::Map<const colvec_type<uint_t>> ph2_counts_v(
                        ph2_counts_.data() + outer_.strides_[j] -
                            outer_.strides_[1],
                        outer_.strides_[j + 1] - outer_.strides_[j]);
                    max_count =
                        std::max(static_cast<int64_t>(prev_count),
                                 static_cast<int64_t>(ph2_counts_v(bits_i[j])));
                    a_star = (max_count != prev_count) ? j : a_star;
                }

                // Phase III

                size_t rej = 0;

                // if current gridpt is regular, do an optimized routine.
                if (gr_view.is_regular(i)) {
                    if (gr_view.check_null(pos, a_star - 1)) {
                        rej = phase_III_internal(a_star, bits_i);
                    }
                    rej_len[pos] = rej;
                    ++pos;
                    continue;
                }

                // else, do a slightly different routine:
                // compute the ph3 test statistic first and loop through each
                // tile to check if it's a false rejection.
                bool rej_computed = false;
                const auto n_ts = gr_view.n_tiles(i);
                for (size_t n_t = 0; n_t < n_ts; ++n_t, ++pos) {
                    bool is_null = gr_view.check_null(pos, a_star - 1);
                    if (!rej_computed && is_null) {
                        rej = phase_III_internal(a_star, bits_i);
                        rej_computed = true;
                    }
                    rej_len[pos] = is_null ? rej : 0;
                }
            }
        }

        /*
         * Computes the gradient of the log-likelihood ratio for arm "arm" and
         * gridpoint "gridpt": T - \nabla_\eta A(\eta) where T is the sufficient
         * statistic (vector), A is the log-partition function, and \eta is the
         * natural parameter.
         *
         * @param   gridpt      gridpoint index.
         * @param   arm         arm index.
         */
        value_t grad(uint_t gridpt, uint_t arm) override {
            auto& bits = outer_.gbits_;
            auto p_ = outer_.p_();
            Eigen::Map<const colvec_type<uint_t>> ss_a(
                suff_stat_.data() + outer_.strides_(arm),
                outer_.strides_(arm + 1) - outer_.strides_(arm));
            return ss_a(bits(arm, gridpt)) -
                   outer_.n_samples() * p_(arm, gridpt);
        }

       private:
        template <class BitsType>
        KEVLAR_STRONG_INLINE auto phase_III_internal(size_t a_star,
                                                     BitsType& bits_i) {
            // pairwise z-test
            auto n = outer_.n_samples();
            Eigen::Map<const colvec_type<uint_t>> ss_astar(
                suff_stat_.data() + outer_.strides_[a_star],
                outer_.strides_[a_star + 1] - outer_.strides_[a_star]);
            Eigen::Map<const colvec_type<uint_t>> ss_0(suff_stat_.data(),
                                                       outer_.strides_[1]);
            auto p_star = static_cast<value_t>(ss_astar(bits_i[a_star])) / n;
            auto p_0 = static_cast<value_t>(ss_0(bits_i[0])) / n;
            auto z = (p_star - p_0);
            auto var = (p_star * (1. - p_star) + p_0 * (1. - p_0));
            z = (var <= 0) ? std::copysign(1.0, z) *
                                 std::numeric_limits<value_t>::infinity()
                           : z / std::sqrt(var / n);

            int i = 0;
            for (; i < outer_.thresholds_.size(); ++i) {
                if (z > outer_.thresholds_[i]) break;
            }

            return outer_.n_models() - i;
        };

        std::uniform_real_distribution<value_t> unif_dist_;
        mat_type<value_t> unif_;         // uniform rng
        colvec_type<uint_t> suff_stat_;  // sufficient statistic table for each
                                         // arm and prob value suff_stat_(i,j) =
                                         // suff stat at unique prob i at arm j.
        colvec_type<uint_t>
            ph2_counts_;  // sufficient statistic table only looking at phase 2
                          // and treatment arms ph2_counts_(i,j) = ph2 suff stat
                          // at unique prob i at arm j.
        size_t n_total_uniques_ = 0;
    };

    using state_t = StateType;

    // default constructor is the base constructor
    using base_t::base_t;

    using base_t::n_arms;
    using base_t::n_samples;
    using base_t::ph2_size;

    /*
     * Constructs the model object with configuration parameters.
     *
     * @param   n_arms      number of arms.
     * @param   ph2_size    phase II size.
     * @param   n_samples   number of patients in each arm.
     * @param   thresholds  critical thresholds.
     *                      Internally, a copy is made in decreasing order.
     */
    BinomialControlkTreatment(
        size_t n_arms, size_t ph2_size, size_t n_samples,
        const Eigen::Ref<const colvec_type<value_t>>& thresholds)
        : base_t(n_arms, ph2_size, n_samples) {
        set_thresholds(thresholds);
    }

    /*
     * Sets the grid range and caches any results
     * to speed-up the simulations.
     */
    void set_grid_range(const grid_range_t& grid_range) {
        model_base_t::set_grid_range(grid_range);

        // resize all other internal quantities
        probs_unique_.resize(n_arms());
        strides_.resize(n_arms() + 1);
        probs_.resize(n_arms() * grid_range.n_gridpts() * 3);
        gbits_.resize(n_arms(), grid_range.n_gridpts());

        auto& thetas = grid_range.thetas();

        // populate prob matrix
        auto& radii = grid_range.radii();
        Eigen::Map<mat_type<value_t>> pm(probs_.data(), n_arms(),
                                         grid_range.n_gridpts());
        pm = thetas - radii;
        new (&pm) Eigen::Map<mat_type<value_t>>(pm.data() + pm.size(),
                                                pm.rows(), pm.cols());
        pm = thetas;
        new (&pm) Eigen::Map<mat_type<value_t>>(pm.data() + pm.size(),
                                                pm.rows(), pm.cols());
        pm = thetas + radii;
        probs_.array() = sigmoid(probs_.array());

        // populate set of unique theta values for each arm
        std::unordered_map<value_t, uint_t> pu_to_idx;
        std::set<value_t> prob_set;
        auto& bits = gbits_;
        auto prob_view = p_();

        strides_[0] = 0;  // initialize stride to arm 0.

        for (size_t i = 0; i < n_arms(); ++i) {
            pu_to_idx.clear();
            prob_set.clear();

            // insert all prob values in arm i into the set
            auto prob_row = prob_view.row(i);
            for (int j = 0; j < prob_row.size(); ++j) {
                prob_set.insert(prob_row[j]);
            }

            // create a mapping from unique prob values to order idx
            {
                int j = 0;
                for (auto p : prob_set) {
                    pu_to_idx[p] = j++;
                }
            }

            // copy number of uniques
            strides_[i + 1] = strides_[i] + prob_set.size();

            // copy unique prob values into vector
            probs_unique_[i].resize(prob_set.size());
            std::copy(prob_set.begin(), prob_set.end(),
                      probs_unique_[i].data());

            // populate bits for current arm
            auto bits_i = bits.row(i);
            for (int j = 0; j < bits_i.size(); ++j) {
                bits_i(j) = pu_to_idx[prob_view(i, j)];
            }
        }
    }

    /*
     * Create a state object associated with the current model instance.
     */
    std::unique_ptr<typename state_t::model_state_base_t> make_state()
        const override {
        return std::make_unique<state_t>(*this);
    }

    uint_t n_models() const override { return thresholds_.size(); }

    /*
     * Computes the quadratic form of covariance matrix
     *  v^T Cov v
     * at jth gridpoint.
     *
     * @param   j      jth gridpoint.
     * @param   v      vector to take quadratic form.
     */
    value_t cov_quad(size_t j, const Eigen::Ref<const colvec_type<value_t>>& v)
        const override {
        auto p_j = p_().col(j);
        return n_samples() * v.array().square().matrix().dot(
                                 (p_j.array() * (1.0 - p_j.array())).matrix());
    }

    /*
     * Computes the supremum (in the grid) of covariance matrix
     *  v^T max_{grid} C v
     * at jth gridpoint.
     *
     * @param   j      jth gridpoint.
     * @param   v      vector to take quadratic form.
     */
    value_t max_cov_quad(
        size_t j,
        const Eigen::Ref<const colvec_type<value_t>>& v) const override {
        auto p = p_();

        Eigen::Map<const mat_type<value_t>> p_lower(probs_.data(), p.rows(),
                                                    p.cols());
        Eigen::Map<const mat_type<value_t>> p_upper(p.data() + p.size(),
                                                    p.rows(), p.cols());
        auto pli = p_lower.col(j);
        auto pui = p_upper.col(j);

        value_t hess_bd = 0;
        for (int k = 0; k < v.size(); ++k) {
            auto v_sq = v[k] * v[k];
            if (pli[k] <= 0.5 && 0.5 <= pui[k]) {
                hess_bd += 0.25 * v_sq;
            } else {
                auto lower = pli[k] - 0.5;  // shift away center
                auto upper = pui[k] - 0.5;  // shift away center
                // max of p(1-p) occurs for whichever p is closest to 0.5.
                bool max_at_upper = (std::abs(upper) < std::abs(lower));
                auto max_endpt = max_at_upper ? pui[k] : pli[k];
                hess_bd += max_endpt * (1. - max_endpt) * v_sq;
            }
        }
        return hess_bd * n_samples();
    }

    /*
     * Set the critical thresholds.
     */
    void set_thresholds(const Eigen::Ref<const colvec_type<value_t>>& thrs) {
        thresholds_ = thrs;
        std::sort(thresholds_.data(), thresholds_.data() + thresholds_.size(),
                  std::greater<value_t>());
    }

    /*
     * Sets the internal structure with the parameters.
     * Users should not interact with this method.
     * It is exposed purely for internal purposes (pickling).
     */
    void set_internal(const std::vector<colvec_type<value_t>>& probs_unique,
                      const colvec_type<uint_t>& strides,
                      const colvec_type<value_t>& probs,
                      const mat_type<uint_t>& gbits) {
        probs_unique_ = probs_unique;
        strides_ = strides;
        probs_ = probs;
        gbits_ = gbits;
    }

    /* Getter routines mainly for pickling */
    const auto& probs_unique() const { return probs_unique_; }
    const auto& strides() const { return strides_; }
    const auto& probs() const { return probs_; }
    const auto& gbits() const { return gbits_; }
    const auto& thresholds() const { return thresholds_; }

   private:
    constexpr auto n_gridpts() const { return gbits_.cols(); }

    auto p_() {
        return Eigen::Map<mat_type<value_t>>(
            probs_.data() + n_arms() * n_gridpts(), n_arms(), n_gridpts());
    }
    auto p_() const {
        return Eigen::Map<const mat_type<value_t>>(
            probs_.data() + n_arms() * n_gridpts(), n_arms(), n_gridpts());
    }

    using vec_t = colvec_type<value_t>;
    using uvec_t = colvec_type<uint_t>;
    using mat_t = mat_type<value_t>;
    using umat_t = mat_type<uint_t>;

    std::vector<vec_t> probs_unique_;  // probs_unique_[i] = unique prob vector
                                       // sorted (ascending) for arm i.
    uvec_t strides_;  // strides_[i] = number of unique probs for arm i-1 with 0
                      // for arm -1.
    vec_t probs_;   // probs_(.,j,k) = jth prob vector with k = 0,1,2 corresp to
                    // left/curr/right gridpt.
    umat_t gbits_;  // range of gbits
    vec_t thresholds_;  // critical thresholds
};

}  // namespace kevlar
