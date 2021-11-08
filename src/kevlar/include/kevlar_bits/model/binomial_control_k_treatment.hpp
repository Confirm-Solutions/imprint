#pragma once
#include <kevlar_bits/util/types.hpp>
#include <kevlar_bits/util/algorithm.hpp>
#include <kevlar_bits/util/d_ary_int.hpp>
#include <kevlar_bits/util/math.hpp>
#include <kevlar_bits/util/hardware.hpp>
#include <kevlar_bits/model/base.hpp>
#include <Eigen/Core>
#include <limits>
#include <algorithm>

namespace kevlar {

/* Forward declaration */
template <class GridType = grid::Arbitrary>
struct BinomialControlkTreatment;

namespace internal {

struct BinomialControlkTreatmentBase
{
    BinomialControlkTreatmentBase(
            size_t n_arms,
            size_t ph2_size,
            size_t n_samples
            )
        : n_arms_(n_arms)
        , ph2_size_(ph2_size)
        , n_samples_(n_samples)
    {}

protected:
    size_t n_arms_;
    size_t ph2_size_;
    size_t n_samples_;
};

template <class T>
struct traits;

template <class GridType>
struct traits<BinomialControlkTreatment<GridType> >
{
    using upper_bd_t = typename BinomialControlkTreatment<GridType>::UpperBound;
};

} // namespace internal

/*
 * Binomial control + k Treatment model.
 * For a given point null p = (p_0,..., p_{k}),
 * n responses Y_{ij} for each arm j=0,...,k where Y_{ij} ~ Bern(p_j) iid,
 * Phase II size of ph2_size, it does the following procedure:
 *
 *  - select the treatment arm j* with most responses based on the first ph2_size samples
 *  - construct the paired z-test between p_{j*} and p_0 testing for the null that p_{j*} <= p_0.
 */

// ========================================================
// BinomialControlkTreatment DECLARATIONS
// ========================================================

/* Specialization declaration: arbitrary grid */
template <>
struct BinomialControlkTreatment<grid::Arbitrary>
    : internal::BinomialControlkTreatmentBase
{
private:
    using base_t = internal::BinomialControlkTreatmentBase;

public:
    struct UpperBound;

    using upper_bd_t = UpperBound;
    using base_t::base_t;

    /*
     * Runs the Binomial 3-arm Phase II/III trial simulation.
     * 
     * @param   unif            matrix with n_arms columns where column i is the uniform draws of arm i.
     * @param   p_range         current range of the full p-grid.
     * @param   thr_grid        Grid of threshold values for tuning. See requirements for UpperBound.
     * @param   upper_bd        Upper-bound object to update.
     */
    template <class UnifType
            , class PRangeType
            , class ThrVecType>
    inline void run(
            const UnifType& unif,
            const PRangeType& p_range,
            const ThrVecType& thr_grid,
            upper_bd_t& upper_bd
            ) const;

    /*
     * Creates an upper bound object associated with the metadata of the current object.
     */
    inline auto make_upper_bd() const;
};

/* Specialization declaration: rectangular grid */
template <>
struct BinomialControlkTreatment<grid::Rectangular>
    : internal::BinomialControlkTreatmentBase
    , ModelBase<BinomialControlkTreatment<grid::Rectangular> >
{
private:
    using base_t = internal::BinomialControlkTreatmentBase;

public:
    struct UpperBound;

    using upper_bd_t = UpperBound;
    using base_t::base_t;

    template <class UnifType
            , class PRangeType
            , class ThrVecType>
    inline void run(
            const UnifType& unif,
            const PRangeType& p_range,
            const ThrVecType& thr_grid,
            upper_bd_t& upper_bd
            ) const;

    inline auto make_upper_bd() const;
};

// ========================================================
// UpperBound DEFINITIONS                                           
// ========================================================

/* Definition of UpperBound nested class: arbitrary grid */
struct BinomialControlkTreatment<grid::Arbitrary>::UpperBound
{
    using outer_t = BinomialControlkTreatment<grid::Arbitrary>;

    UpperBound(const outer_t& outer)
        : outer_{outer}
    {}

    // TODO: member fns
    
private:
    const outer_t& outer_;
};

/* Definition of UpperBound nested class: rectangular grid */
struct BinomialControlkTreatment<grid::Rectangular>::UpperBound
{
private:
    auto upper_bd_constant(double width) const
    {
        return (width * (upper_bd_.array() * (1. - upper_bd_.array()) / n_).sqrt()).matrix();
    }


public:
    using outer_t = BinomialControlkTreatment<grid::Rectangular>;

    UpperBound(const outer_t& outer)
        : outer_{outer}
    {}

    /*
     * Beginning at the grid point index defined by p_idxer,
     * and ending upper_bd_.cols() number of coordinates later,
     * the rejection proportions are updated as running averages in upper_bd_
     * and the gradient components are updated as running averages in grad_buff_.
     *
     * This operation puts the state into Createable.
     * Assumes that the internals have been initialized properly (e.g. see reset()).
     *
     * @param   thr_vec     vector of thresholds. Must be in decreasing order.
     *                      See requirements for create().
     */
    template <class PRangeType
            , class SuffStatType
            , class ThrVecType
            , class TestStatFType>
    void update(
            const PRangeType& p_range,
            SuffStatType& suff_stat,
            const ThrVecType& thr_vec,
            TestStatFType test_stat_f
            )
    {
        using value_t = typename std::decay_t<ThrVecType>::Scalar;
        using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;

        ++n_;

        auto p_begin = p_range.begin();

        for (int j = 0; j < upper_bd_.cols(); ++j, ++p_begin) {
            auto& p_idxer = *p_begin;
            auto test_stat = test_stat_f(p_idxer);

            // find first threshold s.d. test_stat > thr
            auto begin = thr_vec.data();
            auto end = begin + thr_vec.size();
            auto it = std::upper_bound(begin, end, test_stat, std::greater<value_t>());

            // update rejection proportion only in the rows that reject
            auto rej_length = std::distance(it, end);
            auto upper_bd_j = upper_bd_.col(j);
            upper_bd_j.tail(rej_length).array() += 1;

            // update gradient for each dimension
            const auto slice_size = upper_bd_.size();
            auto slice_offset = 0;
            auto& p_idxer_bits = p_idxer();
            auto& p = p_begin.get_1d_grid();
            for (size_t k = 0; k < outer_.n_arms_; ++k, slice_offset += slice_size) {
                Eigen::Map<mat_t> grad_k_cache(
                        grad_buff_.data() + slice_offset, 
                        upper_bd_.rows(),
                        upper_bd_.cols());
                auto grad_k_j = grad_k_cache.col(j);
                // add new_factor * (x_k / n - p_k) for each threshold where we have rejection.
                grad_k_j.tail(rej_length).array() += 
                    static_cast<double>(suff_stat(p_idxer_bits[k], k)) / outer_.n_samples_ 
                     - p[p_idxer_bits[k]];
            }
        }
    }

    /*
     * Pools other upper bound objects into the current object.
     * Must be in state Createable to be a valid call.
     *
     * @param   other       other UpperBound object to pool into current object.
     */
    void pool(const UpperBound& other)
    {
        upper_bd_ += other.upper_bd_;
        grad_buff_ += other.grad_buff_;
        n_ += other.n_;
    }

    /*
     * Creates a full upper bound estimate.
     * Must be in state Createable to be a valid call.
     * After the call, the state is not in Createable.
     *
     * @param   p_idxer     a d-ary integer that will index the vector p to get the configuration of the grid point.
     * @param   p           vector of 1-d p values. Algorithm is only statistically valid when
     *                      p was constructed from an evenly-spaced of radius grid_radius in the natural parameter space.
     * @param   p_endpt     a 2 x c matrix where c is the length of p.
     *                      Each column c contains the endpoints of the 1-d grid centered at p[c].
     *                      The first row must be element-wise less than the second row.
     * @param   alpha       Nominal level of test.
     * @param   width       The constant term upper bound width (width * standard_error / sqrt(n_)).
     *                      Usually, this will be some critical threshold for 1-delta/2 quantile of some asymptotic distribution (normal)
     *                      where delta is the maximum probability of the upper bound exceeding alpha level.
     * @param   grid_radius radius of the grid in the natural parameter space.
     */
    template <class PRangeType, class PEndPtType>
    void create(const PRangeType& p_range, 
                const PEndPtType& p_endpt,
                double alpha,
                double width, 
                double grid_radius)
    {
        // divide to get true averages
        upper_bd_ /= n_;
        grad_buff_ *= static_cast<double>(outer_.n_samples_) / n_;

        // add upper bound for constant term
        upper_bd_ += upper_bd_constant(width);

        // add epsilon * ||grad term||_L^1
        const auto slice_size = upper_bd_.size();
        auto slice_offset = 0;
        for (size_t k = 0; k < outer_.n_arms_; ++k, slice_offset += slice_size) {
            Eigen::Map<mat_t> grad_k(
                    grad_buff_.data() + slice_offset,
                    upper_bd_.rows(),
                    upper_bd_.cols());
            upper_bd_.array() += grid_radius * grad_k.array().abs();
        }

        // add upper bound for gradient term and hessian term
        upper_bd_grad_hess(
            p_range, p_endpt, alpha, grid_radius,
            [&](Eigen::Index i, auto grad_bd, auto hess_bd) {
                upper_bd_.col(i).array() += grad_bd + hess_bd;
            });

    }

    /* 
     * Resets the internals to consider m number of thresholds 
     * and n number of grid points when updating.
     */
    void reset(size_t m, size_t n)
    {
        upper_bd_.setZero(m, n);
        grad_buff_.setZero(m * n * outer_.n_arms_);
        n_ = 0;
    }

    auto& get() { return upper_bd_; }
    const auto& get() const { return upper_bd_; }

    /*
     * Computes a hint for what the batch size of parameters should be.
     * Given the number of thresholds (thr_vec_size) to consider,
     * we estimate this from a pre-fit GLM model with hard-coded constants.
     */
    auto p_batch_size_hint(size_t thr_vec_size) const 
    {
        // Heuristic for guessing how much can fit in the cache:
        // Performance seems pretty good according to this formula.
        size_t out = 7430 * std::exp(-0.035717058454485931 * thr_vec_size);
        return out;
    }

    /*
     * Serializes the necessary quantities to construct the upper bound.
     * Assumes that the object is in state Createable.
     */
    template <class SerializerType
            , class PRangeType
            , class PEndptType>
    void serialize(SerializerType& s, 
                   const PRangeType& p_range,
                   const PEndptType& p_endpt,
                   double alpha,
                   double width,
                   double grid_radius) 
    {
        auto& p = p_range.get_1d_grid();

        if (!serialized_) {
            uint32_t n_total = ipow(p.size(), outer_.n_arms_);
            uint32_t n_arms = outer_.n_arms_;
            s << n_total << n_arms;
            serialized_ = true;
        }

        upper_bd_ /= n_;                
        grad_buff_ *= static_cast<double>(outer_.n_samples_) / n_;

        // serialize 1/N sum_{i=1}^N 1_{rej hyp i}
        s << upper_bd_;

        // replace the matrix with upper bound for constant term
        upper_bd_.array() = upper_bd_constant(width);
        s << upper_bd_;

        // serialize gradient (for all components)
        s << grad_buff_;

        // add upper bound for gradient term and hessian term
        upper_bd_grad_hess(
                p_range, p_endpt, alpha, grid_radius,
                [&](Eigen::Index, auto grad_bd, auto hess_bd) {
                    s << grad_bd << hess_bd;
                });
    }

    /*
     * Unserializes and stores the results into the arguments.
     * TODO: this is only a valid operation if the Unserializer is processing a file
     * that was created from calling serialize() with 1-threshold considered.
     *
     * @param   c_vec       vector for constant monte carlo estimate (p).
     * @param   c_bd_vec    vector for constant upper bound (p).
     * @param   grad_mat    matrix for gradient monte carlo estimates for all arms (p x k).
     * @param   grad_bd_vec vector for gradient upper bound (p).
     * @param   hess_bd_vec vector for hessian upper bound (p).
     */
    template <class UnSerializerType
            , class ConstantVecType
            , class ConstantBdVecType
            , class GradMatType
            , class GradBdVecType
            , class HessBdVecType>
    static void unserialize(
            UnSerializerType& us,
            ConstantVecType& c_vec,
            ConstantBdVecType& c_bd_vec,
            GradMatType& grad_mat,
            GradBdVecType& grad_bd_vec,
            HessBdVecType& hess_bd_vec
            ) 
    {
        uint32_t n_total, n_arms;
        us >> n_total >> n_arms;

        c_vec.resize(n_total);
        c_bd_vec.resize(n_total);
        grad_mat.resize(n_total, n_arms);
        grad_bd_vec.resize(n_total);
        hess_bd_vec.resize(n_total);

        size_t offset = 0;

        Eigen::MatrixXd cache;
        Eigen::VectorXd buff;

        while (us.get()) {

            // read batch of constant matrix
            us >> cache;
            Eigen::Map<Eigen::MatrixXd> viewer(
                    c_vec.data() + offset,
                    cache.rows(),
                    cache.cols()
                    );
            viewer = cache;

            // read batch of constant upper bd matrix
            us >> cache;
            new (&viewer) Eigen::Map<Eigen::MatrixXd>(
                    c_bd_vec.data() + offset,
                    cache.rows(),
                    cache.cols()
                    );
            viewer = cache;

            // read batch of gradient vector
            us >> buff;
            new (&viewer) Eigen::Map<Eigen::VectorXd>(
                    buff.data(),
                    cache.size(),
                    grad_mat.cols()
                    );
            grad_mat.block(offset, 0, cache.size(), grad_mat.cols())
                = viewer;

            // read batch of gradient/hessian bounds
            Eigen::Map<Eigen::VectorXd> grad_bd_viewer(
                    grad_bd_vec.data() + offset,
                    cache.cols()
                    );
            Eigen::Map<Eigen::VectorXd> hess_bd_viewer(
                    hess_bd_vec.data() + offset,
                    cache.cols()
                    );
            for (int i = 0; i < cache.cols(); ++i) {
                us >> grad_bd_viewer[i];
                us >> hess_bd_viewer[i];
            }

            // update offset
            offset += cache.size();
        }
    }

private:

    template <class PRangeType
            , class PEndptType
            , class FType>
    void upper_bd_grad_hess(
            const PRangeType& p_range,
            const PEndptType& p_endpt,
            double alpha,
            double grid_radius,
            FType f) const 
    {
        auto p_begin = p_range.begin();

        for (Eigen::Index i = 0; i < upper_bd_.cols(); ++i, ++p_begin) {
            
            auto& p_idxer = *p_begin;
            auto& bits = p_idxer();
            auto& p = p_begin.get_1d_grid();

            // compute grad upper bound
            double var = 0;
            std::for_each(bits.data(), bits.data() + bits.size(),
                [&](auto k) { var += p[k] * (1.-p[k]); });
                
            var *= (static_cast<double>(outer_.n_samples_) / n_) * (1./alpha - 1.);
            double grad_bd = grid_radius * std::sqrt(var);

            // compute hessian upper bound
            double hess_bd = 0;
            std::for_each(bits.data(), bits.data() + bits.size(),
                [&](auto k) {
                    auto col_k = p_endpt.col(k);
                    auto lower = col_k[0] - 0.5; // shift away center
                    auto upper = col_k[1] - 0.5; // shift away center
                    // max of p(1-p) occurs for whichever p is closest to 0.5.
                    bool max_at_upper = (std::abs(upper) < std::abs(lower));
                    auto max_endpt = col_k[max_at_upper]; 
                    hess_bd += max_endpt * (1. - max_endpt);
                });
            hess_bd *= (outer_.n_samples_ * grid_radius * grid_radius) / 2.;
            f(i, grad_bd, hess_bd);
        }
    }

    using mat_t = Eigen::MatrixXd;

    // This matrix will be used to store the final upper bound.
    // To save memory, it will be used to store the rejection proportions during update.
    mat_t upper_bd_;
    Eigen::VectorXd grad_buff_;
    const outer_t& outer_;
    size_t n_ = 0;  // number of updates
    bool serialized_ = false;   // true iff serialize() has been called.
};

// ========================================================
// BinomialControlkTreatment Run DEFINITIONS
// ========================================================

/* Specialization: arbitrary */
template <class UnifType
        , class PRangeType
        , class ThrVecType>
inline void BinomialControlkTreatment<grid::Arbitrary>::run(
            const UnifType& unif,
            const PRangeType& p_range,
            const ThrVecType& thr_grid,
            upper_bd_t& upper_bd
            ) const
{
    assert(static_cast<size_t>(unif.rows()) == n_samples_);
    assert(static_cast<size_t>(unif.cols()) == n_arms_);

    size_t k = unif.cols()-1;
    size_t n = unif.rows();
    size_t ph2_size = ph2_size;
    size_t ph3_size = n - ph2_size;

    // resize cache
    Eigen::VectorXd a_sum(k);

    auto z_stat = [&](const auto& p) {
        // phase II
        for (size_t i = 0; i < a_sum.size(); ++i) {
            a_sum[i] = (unif.col(i+1).head(ph2_size).array() < p(i+1)).count();
        }

        // compare and choose arm with more successes
        Eigen::Index a_star;
        a_sum.maxCoeff(&a_star);

        // phase III
        size_t a_star_rest_sum = (unif.col(a_star+1).tail(ph3_size).array() < p(a_star+1)).count();
        auto p_star = static_cast<double>(a_sum[a_star] + a_star_rest_sum) / n;
        auto p_0 = (unif.col(0).array() < p(0)).template cast<double>().mean();
        auto z = (p_star - p_0);
        auto var = (p_star * (1.-p_star) + p_0 * (1.-p_0));
        z = (var == 0) ? std::numeric_limits<double>::infinity() : z / sqrt(var / n); 
        return z;
    };
}

/* Specialization: rectangular */
template <class UnifType
        , class PRangeType
        , class ThrVecType>
inline void 
BinomialControlkTreatment<grid::Rectangular>::run(
        const UnifType& unif,
        const PRangeType& p_range,
        const ThrVecType& thr_grid,
        upper_bd_t& upper_bd
        ) const
{
    // uniform should have the expected shape
    assert(static_cast<size_t>(unif.rows()) == n_samples_);
    assert(static_cast<size_t>(unif.cols()) == n_arms_);

    size_t k = unif.cols()-1;
    size_t n = unif.rows();
    size_t ph2_size = ph2_size_;
    size_t ph3_size = n - ph2_size;
    auto& p = p_range.get_1d_grid();
    size_t d = p.size();

    Eigen::VectorXd p_sorted = p;
    std::sort(p_sorted.data(), p_sorted.data() + d);

    // TODO: maybe optimization? no need for copy?
    Eigen::MatrixXd ph2_unif = unif.block(0, 1, ph2_size, k);
    Eigen::MatrixXd ph3_unif = unif.block(ph2_size, 1, ph3_size, k);
    Eigen::VectorXd control_unif = unif.col(0);
    sort_cols(ph2_unif);
    sort_cols(ph3_unif);
    sort_cols(control_unif);

    Eigen::MatrixXi suff_stat(d, k+1);
    Eigen::Map<Eigen::VectorXi> control_counts(suff_stat.data(), d);
    Eigen::Map<Eigen::MatrixXi> ph2_counts(suff_stat.col(1).data(), d, k);
    Eigen::MatrixXi ph3_counts(d, k);
    cum_count(ph2_unif, p_sorted, ph2_counts);
    cum_count(ph3_unif, p_sorted, ph3_counts);
    cum_count(control_unif, p_sorted, control_counts);
    suff_stat.block(0, 1, d, k) += ph3_counts;

    auto z_stat = [&](const auto& p_idxer) {
        auto& idx = p_idxer();

        // Phase II
        int a_star = -1;
        int max_count = -1;
        for (int j = 1; j < idx.size(); ++j) {
            int prev_count = max_count;
            max_count = std::max(prev_count, ph2_counts(idx[j], j-1));
            a_star = (max_count != prev_count) ? j : a_star;
        }

        // Phase III
        auto p_star = static_cast<double>(suff_stat(idx[a_star], a_star)) / n;
        auto p_0 = static_cast<double>(control_counts(idx[0])) / n;
        auto z = (p_star - p_0);
        auto var = (p_star * (1.-p_star) + p_0 * (1.-p_0));
        z = (var <= 0) ? std::numeric_limits<double>::infinity() : z / std::sqrt(var / n); 

        return z;
    };

    upper_bd.update(p_range, suff_stat, thr_grid, z_stat);
}

// ========================================================
// make_upper_bd DEFINITIONS
// ========================================================
#define MAKE_UPPER_BD_GEN(grid_type) \
    inline auto \
    BinomialControlkTreatment<grid::grid_type>::make_upper_bd() const \
    { return UpperBound(*this); } 

MAKE_UPPER_BD_GEN(Arbitrary)
MAKE_UPPER_BD_GEN(Rectangular)

#undef MAKE_UPPER_BD_GEN

} // namespace kevlar
