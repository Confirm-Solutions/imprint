#pragma once
#include <kevlar_bits/util/types.hpp>
#include <kevlar_bits/util/algorithm.hpp>
#include <kevlar_bits/util/d_ary_int.hpp>
#include <kevlar_bits/util/math.hpp>
#include <Eigen/Core>
#include <limits>
#include <algorithm>

namespace kevlar {

template <class GridType = grid::Arbitrary>
struct BinomialControlkTreatment
{
    /*
     * Runs the Binomial 3-arm phase II/III trial simulation.
     * 
     * @param   p           vector of length k with probability of each arm success rate (null point).
     * @param   unif        matrix with k columns where column i is the uniform draws of arm i.
     * @param   ph2_size    phase ii number of patients from treatment arms.
     *                      Assumes the treatment arms have the same number of patients.
     */
    template <class PType, class UnifType>
    auto run(
        const PType& p, 
        const UnifType& unif,
        size_t ph2_size)
    {
        size_t k = unif.cols()-1;
        size_t n = unif.rows();
        size_t ph3_size = n - ph2_size;

        // resize cache
        a_sum_.conservativeResize(k);

        // phase II
        for (size_t i = 0; i < k; ++i) {
            a_sum_[i] = (unif.col(i+1).head(ph2_size).array() < p(i+1)).count();
        }

        // compare and choose arm with more successes
        Eigen::Index a_star;
        a_sum_.maxCoeff(&a_star);

        // phase III
        size_t a_star_rest_sum = (unif.col(a_star+1).tail(ph3_size).array() < p(a_star+1)).count();
        auto p_star = static_cast<double>(a_sum_[a_star] + a_star_rest_sum) / n;
        auto p_0 = (unif.col(0).array() < p(0)).template cast<double>().mean();
        auto z = (p_star - p_0);
        auto var = (p_star * (1.-p_star) + p_0 * (1.-p_0));
        z = (var == 0) ? std::numeric_limits<double>::infinity() : z / sqrt(var / n); 
        return z;
    }

private:
    Eigen::VectorXd a_sum_;
};

/* Specialization: Rectangular grid */
template <>
struct BinomialControlkTreatment<grid::Rectangular>
{
    struct UpperBound;

    using upper_bd_t = UpperBound;

    BinomialControlkTreatment(
            size_t n_arms,
            size_t ph2_size,
            size_t n_samples
            )
        : n_arms_(n_arms), ph2_size_(ph2_size), n_samples_(n_samples)
    {}

    /*
     * Runs the Binomial 3-arm Phase II/III trial simulation.
     * 
     * @param   unif            matrix with n_arms columns where column i is the uniform draws of arm i.
     * @param   p_idxer         current indexer for the full p-grid.
     * @param   p_batch_size    number of grid points to consider starting at p_idxer
     * @param   p               vector of length d with probability of an arm success rate (null point).
     *                          Assumes the grid is rectangular with points formed as a tuple of the values in p.
     * @param   thr_grid        Grid of threshold values for tuning. See requirements for UpperBound.
     * @param   upper_bd        Upper-bound object to update.
     */
    template <class UnifType
            , class PType
            , class ThrVecType>
    inline void run(
            const UnifType& unif,
            const dAryInt& p_idxer,
            size_t p_batch_size,
            const PType& p,
            const ThrVecType& thr_grid,
            upper_bd_t& upper_bd
            ) const;

    inline auto make_upper_bd() const;

private:
    size_t n_arms_;
    size_t ph2_size_;
    size_t n_samples_;
};

struct BinomialControlkTreatment<grid::Rectangular>::UpperBound
{
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
     * Assumes that upper_bd_ has already been reshaped and initialized (or updated from previous call).
     *
     * @param   thr_vec     vector of thresholds. Must be in decreasing order.
     *                      See requirements for create().
     */
    template <class PType
            , class SuffStatType
            , class ThrVecType
            , class TestStatFType>
    void update(
            dAryInt p_idxer,
            const PType& p,
            SuffStatType& suff_stat,
            const ThrVecType& thr_vec,
            TestStatFType test_stat_f
            )
    {
        using value_t = typename std::decay_t<ThrVecType>::Scalar;
        using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;

        ++n_;

        // compute new factor for running average
        auto new_factor = static_cast<value_t>(1) / n_;
        auto old_factor = static_cast<value_t>(1) - new_factor;

        // re-weigh all running average quantities
        upper_bd_ *= old_factor;
        grad_buff_ *= old_factor;

        for (int j = 0; j < upper_bd_.cols(); ++j, ++p_idxer) {
            auto test_stat = test_stat_f(p_idxer);

            // find first threshold s.d. test_stat > thr
            auto begin = thr_vec.data();
            auto end = begin + thr_vec.size();
            auto it = std::upper_bound(begin, end, test_stat, std::greater<value_t>());

            // update rejection proportion only in the rows that reject
            auto rej_length = std::distance(it, end);
            auto upper_bd_j = upper_bd_.col(j);
            upper_bd_j.tail(rej_length).array() += new_factor;

            // update gradient for each dimension
            const auto slice_size = upper_bd_.size();
            auto slice_offset = 0;
            auto& p_idxer_bits = p_idxer();
            for (size_t k = 0; k < outer_.n_arms_; ++k, slice_offset += slice_size) {
                Eigen::Map<mat_t> grad_k_cache(
                        grad_buff_.data() + slice_offset, 
                        upper_bd_.rows(),
                        upper_bd_.cols());
                auto grad_k_j = grad_k_cache.col(j);
                // add new_factor * (x_k - n p_k) for each threshold where we have rejection.
                grad_k_j.tail(rej_length).array() += 
                    new_factor * 
                    (suff_stat(p_idxer_bits[k], k) - outer_.n_samples_ * p[p_idxer_bits[k]]);
            }
        }
    }

    void pool(const UpperBound& other, double factor, size_t n)
    {
        upper_bd_ += factor * other.upper_bd_;
        grad_buff_ += factor * other.grad_buff_;
        n_ += n;
    }

    /*
     * @param   p           vector of 1-d p values. Algorithm is only statistically valid when
     *                      p was constructed from an evenly-spaced of radius grid_radius in the natural parameter space.
     * @param   p_endpt     a 2 x c matrix where c is the length of p.
     *                      Each column c contains the endpoints of the 1-d grid centered at p[c].
     *                      The first row must be element-wise less than the second row.
     */
    template <class PType, class PEndPtType>
    void create(dAryInt p_idxer, 
                const PType& p, 
                const PEndPtType& p_endpt,
                double alpha,
                double width, 
                double grid_radius)
    {
        // add upper bound for constant term
        upper_bd_.array() +=
            width * 
            (upper_bd_.array() * (1. - upper_bd_.array()) / n_).sqrt();

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
        for (Eigen::Index i = 0; i < upper_bd_.cols(); ++i, ++p_idxer) {

            auto& bits = p_idxer();

            // compute grad upper bound
            double var = 0;
            std::for_each(bits.data(), bits.data() + bits.size(),
                [&](auto k) { var += p[k] * (1-p[k]); });
                
            var *= (static_cast<double>(outer_.n_samples_) / n_) * (1./alpha - 1.);
            auto grad_bd = grid_radius * std::sqrt(var);

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
            hess_bd *= (outer_.n_samples_ * grid_radius * grid_radius) / 2;

            upper_bd_.col(i).array() += grad_bd + hess_bd;
        }

    }

    void reset(size_t m, size_t n)
    {
        upper_bd_.setZero(m, n);
        grad_buff_.setZero(m * n * outer_.n_arms_);
        n_ = 0;
    }

    auto& get() { return upper_bd_; }
    const auto& get() const { return upper_bd_; }

private:
    using mat_t = Eigen::MatrixXd;

    // This matrix will be used to store the final upper bound.
    // To save memory, it will be used to store the rejection proportions during update.
    mat_t upper_bd_;
    Eigen::VectorXd grad_buff_;
    const outer_t& outer_;
    size_t n_ = 0;  // number of updates
};

template <class UnifType
        , class PType
        , class ThrVecType>
inline void 
BinomialControlkTreatment<grid::Rectangular>::run(
        const UnifType& unif,
        const dAryInt& p_idxer,
        size_t p_batch_size,
        const PType& p,
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
    cum_count(ph2_unif, p, ph2_counts);
    cum_count(ph3_unif, p, ph3_counts);
    cum_count(control_unif, p, control_counts);
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

    upper_bd.update(p_idxer, p, suff_stat, thr_grid, z_stat);
}

inline auto
BinomialControlkTreatment<grid::Rectangular>::make_upper_bd() const
{ return UpperBound(*this); }

} // namespace kevlar
