#pragma once
#include <imprint_bits/distribution/exponential.hpp>
#include <imprint_bits/model/base.hpp>
#include <imprint_bits/util/macros.hpp>
#include <random>

namespace imprint {
namespace model {
namespace exponential {

/*
 * This class represents the cache for all exponential models
 * with 2 arms and with eta transformation
 *
 *      (\log(\lambda_c), \log(h)) \mapsto (\lambda_c, \lambda_c * h)
 *
 * By definition, exponential models are those that assume the data
 * is drawn from an exponential distribution independently across arms.
 * This class further assumes that each arm has the same, fixed number of
 * samples. This class is intended for models that are easily expressable, or
 * even fully described by, hazard rates rather than hazards themselves.
 */
template <class GenType, class ValueType, class UIntType, class GridRangeType>
struct SimGlobalStateFixedNLogHazardRate
    : SimGlobalStateBase<ValueType, UIntType> {
    struct SimState;

    using base_t = SimGlobalStateBase<ValueType, UIntType>;
    using typename base_t::interface_t;
    using typename base_t::uint_t;
    using typename base_t::value_t;
    using gen_t = GenType;
    using grid_range_t = GridRangeType;

    using sim_state_t = SimState;

   private:
    size_t n_arm_samples_;
    mat_type<value_t>
        buff_;  // buff_(0,j) = lambda of control at jth gridpoint.
                // buff_(1,j) = hazard rate at jth gridpoint.

   protected:
    IMPRINT_STRONG_INLINE
    auto lmda_control(size_t j) const { return buff_(0, j); }

    IMPRINT_STRONG_INLINE
    auto hzrd_rate(size_t j) const { return buff_(1, j); }

    IMPRINT_STRONG_INLINE
    constexpr size_t n_params() const { return 2; }
    IMPRINT_STRONG_INLINE
    constexpr size_t n_arm_samples() const { return n_arm_samples_; }

   public:
    SimGlobalStateFixedNLogHazardRate(size_t n_arm_samples,
                                      const grid_range_t& grid_range)
        : n_arm_samples_(n_arm_samples),
          buff_(grid_range.n_params(), grid_range.n_gridpts()) {
        buff_.array() = grid_range.thetas().array().exp();
    }
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
struct SimGlobalStateFixedNLogHazardRate<GenType, ValueType, UIntType,
                                         GridRangeType>::SimState
    : SimGlobalStateFixedNLogHazardRate::base_t::sim_state_t {
   private:
    using outer_t = SimGlobalStateFixedNLogHazardRate;

   public:
    using base_t = typename outer_t::base_t::sim_state_t;
    using typename base_t::interface_t;

   private:
    using exp_t = distribution::Exponential<value_t>;

    const outer_t& outer_;
    exp_t exp_;  // exponential distribution object
    value_t hzrd_rate_ =
        1;  // current hazard rate parameter for the exponential samples.
    mat_type<value_t>
        exp_randoms_;  // exp_randoms_(i,j) =
                       //      Exp(1) draw for patient i in group j=0 (and
                       //      sorted) Exp(hzrd_rate) draw for patient i in
                       //      group j=1 (and sorted) This class assumes
                       //      scale-invariance in that only the ratio of scales
                       //      matter, so it suffices to save relative
                       //      information.

    mat_type<value_t, 2, 1>
        sufficient_stats_;  // sufficient statistic for each arm
                            // - sum of Exp(1) for group 0 (control)
                            // - sum of Exp(hzrd_rate_) for group 1 (treatment)
    gen_t gen_;

   public:
    SimState(const outer_t& outer, size_t seed)
        : outer_(outer),
          exp_(1.0),
          exp_randoms_(outer.n_arm_samples(), outer.n_params()),
          gen_(seed) {}

    IMPRINT_STRONG_INLINE
    auto control() { return exp_randoms_.col(0); }
    IMPRINT_STRONG_INLINE
    auto control() const { return exp_randoms_.col(0); }
    IMPRINT_STRONG_INLINE
    auto treatment() { return exp_randoms_.col(1); }
    IMPRINT_STRONG_INLINE
    auto treatment() const { return exp_randoms_.col(1); }
    IMPRINT_STRONG_INLINE
    auto hzrd_rate() const { return hzrd_rate_; }

    /*
     * Generates exponential randoms
     * The control arm will be Exp(1) draws of size given in outer class.
     * The treatment arm will be Exp(current hazard rate) draws of size given in
     * outer class. If hazard rate was never explicitly set, it is by default
     * set to 1.
     */
    IMPRINT_STRONG_INLINE
    void generate_data() {
        exp_.sample(outer_.n_arm_samples(), outer_.n_params(), gen_,
                    exp_randoms_);
        if (hzrd_rate_ != 1) exp_randoms_.col(1) *= (1. / hzrd_rate_);
    }

    /*
     * Generates the sufficient statistics, which is
     * the sum of the samples for each arm.
     * The control arm will be sum of Exp(1) draws.
     * The treatment arm will be sum of Exp(current hazard rate) draws.
     * This call is undefined if generate_exponentials was not called before.
     */
    IMPRINT_STRONG_INLINE
    void generate_sufficient_stats() {
        sufficient_stats_ = exp_randoms_.colwise().sum();
    }

    /*
     * Updates internal hazard rate to hzrd_rate_new.
     * This will also update the treatment arm and its sufficient stat.
     * It is undefined behavior if hzrd_rate_new <= 0.
     */
    IMPRINT_STRONG_INLINE
    void update_hzrd_rate(value_t hzrd_rate_new) {
        auto hzrd_rate_ratio = (hzrd_rate_ / hzrd_rate_new);
        treatment() *= hzrd_rate_ratio;
        sufficient_stats_[1] *= hzrd_rate_ratio;
        hzrd_rate_ = hzrd_rate_new;
    }

    void score(size_t gridpt_idx,
               Eigen::Ref<colvec_type<value_t>> out) const override {
        assert(out.size() == outer_.n_params());

        auto lmda_c = outer_.lmda_control(gridpt_idx);
        auto inv_lmda_c = 1. / lmda_c;
        auto hzrd_rate_curr = outer_.hzrd_rate(gridpt_idx);

        mat_type<value_t, 2, 1> lmda;
        lmda[0] = lmda_c;
        lmda[1] = hzrd_rate_curr * lmda_c;
        out.array() = exp_t::score(sufficient_stats_.array() * inv_lmda_c,
                                   outer_.n_arm_samples(), lmda.array());
    }
};

template <class _GridRangeType>
struct ImprintBoundStateFixedNLogHazardRate
    : ImprintBoundStateBase<typename _GridRangeType::value_t> {
    using grid_range_t = _GridRangeType;
    using base_t = ImprintBoundStateBase<typename grid_range_t::value_t>;
    using typename base_t::interface_t;
    using typename base_t::value_t;

   private:
    using exp_t = distribution::Exponential<value_t>;

    const mat_type<value_t, 2, 2> max_cov_;
    const size_t n_arm_samples_;
    const value_t max_eta_hess_cov_;
    const mat_type<value_t> lmdas_;

   public:
    ImprintBoundStateFixedNLogHazardRate(size_t n_arm_samples,
                                         const grid_range_t& grid_range)
        : n_arm_samples_(n_arm_samples),
          max_eta_hess_cov_(3 * std::sqrt(n_arm_samples)),
          lmdas_(grid_range.n_params(), grid_range.n_gridpts()) {
        // temporarily const-cast just to initialize the values
        auto& max_cov_nc_ = const_cast<mat_type<value_t, 2, 2>&>(max_cov_);
        max_cov_nc_.setOnes();
        max_cov_nc_(0, 0) = 2;
        max_cov_nc_ *= n_arm_samples;

        auto& lmdas_nc_ = const_cast<mat_type<value_t>&>(lmdas_);
        lmdas_nc_ = grid_range.thetas();
        lmdas_nc_.row(1) += lmdas_nc_.row(0);
        lmdas_nc_.array() = lmdas_nc_.array().exp();
    }

    /*
     * \begin{align*}
     *      D\eta &=
     *      \begin{bmatrix}
     *          -\lambda_1 & 0 \\
     *          -\lambda_2 & -\lambda_2
     *      \end{bmatrix}
     * \end{align*}
     */
    void apply_eta_jacobian(size_t gridpt_idx,
                            const Eigen::Ref<const colvec_type<value_t>>& v,
                            Eigen::Ref<colvec_type<value_t>> out) override {
        assert(v.size() == n_natural_params());
        assert(out.size() == n_natural_params());
        auto lmdas = lmdas_.col(gridpt_idx);
        mat_type<value_t, 2, 2> deta;
        deta(0, 0) = -lmdas[0];
        deta(0, 1) = 0;
        deta.row(1).array() = -lmdas[1];

        out = deta * v;
    }

    /*
     * Computes the covariance quadratic form of v given by:
     *
     *      v^\top
     *      \begin{align*}
     *          \lambda_1^{-1} & 0 \\
     *          0 & \lambda_2^{-1}
     *      \end{align*}
     *      v
     *
     * where $\lambda$ is the mean parameter at grid-point
     * given by gridpt_idx.
     */
    value_t covar_quadform(
        size_t gridpt_idx,
        const Eigen::Ref<const colvec_type<value_t>>& v) override {
        assert(v.size() == n_natural_params());
        auto lmdas = lmdas_.col(gridpt_idx);
        return exp_t::covar_quadform(n_arm_samples_, lmdas.array(), v.array());
    }

    /*
     * Computes the (convex) upper bound U(v) given by:
     *
     *      n
     *      v^\top
     *      \begin{bmatrix}
     *          2 & 1 \\
     *          1 & 1
     *      \end{bmatrix}
     *      v
     *      +
     *      ||v||^2 (3 \sqrt{n})
     *
     * where n is the number of samples per arm.
     */
    value_t hessian_quadform_bound(
        size_t, size_t,
        const Eigen::Ref<const colvec_type<value_t>>& v) override {
        assert(v.size() == n_natural_params());
        return (v.dot(max_cov_ * v)) + v.squaredNorm() * max_eta_hess_cov_;
    }

    size_t n_natural_params() const override { return 2; }
};

}  // namespace exponential
}  // namespace model
}  // namespace imprint
