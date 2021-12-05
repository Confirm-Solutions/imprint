#pragma once
#include <Eigen/Core>
#include <kevlar_bits/util/math.hpp>

namespace kevlar {

/*
 * This class is a generic object for capturing all the logic of constructing an upper bound.
 * Given an Exponential family model object satisfying a certain interface (see member functions below),
 * it stores all necessary components of the upper bound (zeroth/first order term and zeroth/first/second order upper bound corrections).
 */
template <class ValueType>
struct UpperBound
{
private:
    using value_t = ValueType;

    auto upper_bd_constant(value_t delta) const
    {
        auto width = qnorm(1.-delta/2.);
        return (width * (upper_bd_.array() * (1. - upper_bd_.array()) / n_).sqrt()).matrix();
    }

    template <class ModelType
            , class MeanIdxerRangeType
            , class FType>
    void upper_bd_grad_hess(
            const ModelType& model,
            const MeanIdxerRangeType& mean_idxer_range,
            value_t delta,
            value_t grid_radius,
            FType f) const 
    {
        auto idxer_begin = mean_idxer_range.begin();

        for (Eigen::Index i = 0; i < upper_bd_.cols(); ++i, ++idxer_begin) {
            
            auto& idxer = *idxer_begin;

            // compute grad upper bound
            value_t var = model.tr_cov(idxer);
                
            var *= (2./delta - 1.) / n_;
            value_t grad_bd = grid_radius * std::sqrt(var);

            // compute hessian upper bound
            value_t hess_bd = model.tr_max_cov(idxer);
            hess_bd *= (grid_radius * grid_radius) / 2.;
            f(i, grad_bd, hess_bd);
        }
    }

    constexpr auto n_arms() const { return grad_buff_.size() / upper_bd_.size(); }

public:

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
    template <class ModelType
            , class MeanIdxerRangeType
            , class ThrVecType>
    void update(
            const ModelType& model,
            const MeanIdxerRangeType& mean_idxer_range,
            const ThrVecType& thr_vec
            )
    {
        ++n_;

        auto idxer_begin = mean_idxer_range.begin();

        // iterate over each configuration of model parameters
        for (int j = 0; j < upper_bd_.cols(); ++j, ++idxer_begin) {
            auto& idxer = *idxer_begin;
            auto test_stat = model.test_stat(idxer);

            // find first threshold s.d. test_stat > thr
            auto begin = thr_vec.data();
            auto end = begin + thr_vec.size();
            auto it = std::upper_bound(begin, end, test_stat, std::greater<value_t>());

            // update rejection count only in the rows that reject
            auto rej_length = std::distance(it, end);
            auto upper_bd_j = upper_bd_.col(j);
            upper_bd_j.tail(rej_length).array() += 1;

            // update gradient for each dimension
            const auto slice_size = upper_bd_.size();
            auto slice_offset = 0;
            auto& idxer_bits = idxer();
            size_t n_arms = this->n_arms();
            for (size_t k = 0; k < n_arms; ++k, slice_offset += slice_size) {
                Eigen::Map<mat_t> grad_k_cache(
                        grad_buff_.data() + slice_offset, 
                        upper_bd_.rows(),
                        upper_bd_.cols());
                auto grad_k_j = grad_k_cache.col(j);

                // add (T - nabla_eta A(m)) for each threshold where we have rejection.
                // where T is the sufficient statistic for arm k under mean m,
                // nabla_eta A(m) is the gradient under the natural parameter eta of the log-partition function for arm k evaluated at mean m.
                grad_k_j.tail(rej_length).array() += model.grad_lr(k, idxer_bits[k]);
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
     * @param   grid_radius radius of the grid in the natural parameter space.
     */
    template <class ModelType, class MeanIdxerRangeType>
    void create(const ModelType& model,
                const MeanIdxerRangeType& mean_idxer_range, 
                value_t delta,
                value_t grid_radius)
    {
        // divide to get true averages
        upper_bd_ /= n_;
        grad_buff_ /= n_;

        // add upper bound for constant term
        upper_bd_ += upper_bd_constant(delta);

        // add epsilon * ||grad term||_L^1
        const auto slice_size = upper_bd_.size();
        auto slice_offset = 0;
        size_t n_arms = this->n_arms();
        for (size_t k = 0; k < n_arms; ++k, slice_offset += slice_size) {
            Eigen::Map<mat_t> grad_k(
                    grad_buff_.data() + slice_offset,
                    upper_bd_.rows(),
                    upper_bd_.cols());
            upper_bd_.array() += grid_radius * grad_k.array().abs();
        }

        // add upper bound for gradient term and hessian term
        upper_bd_grad_hess(
            model, mean_idxer_range, delta, grid_radius,
            [&](Eigen::Index i, auto grad_bd, auto hess_bd) {
                upper_bd_.col(i).array() += grad_bd + hess_bd;
            });
    }

    /* 
     * Resets the internals to consider m number of thresholds,
     * n number of grid points when updating, and k arms.
     */
    void reset(size_t m, size_t n, size_t k)
    {
        upper_bd_.setZero(m, n);
        grad_buff_.setZero(m * n * k);
        n_ = 0;
    }

    auto& get() { return upper_bd_; }
    const auto& get() const { return upper_bd_; }

    /*
     * Computes a hint for what the batch size of parameters should be.
     * Given the number of thresholds (thr_vec_size) to consider,
     * we estimate this from a pre-fit GLM model with hard-coded constants.
     */
    constexpr auto p_batch_size_hint(size_t thr_vec_size) const 
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
            , class ModelType
            , class MeanIdxerRangeType>
    void serialize(SerializerType& s, 
                   const ModelType& model,
                   const MeanIdxerRangeType& mean_idxer_range,
                   value_t delta,
                   value_t grid_radius) 
    {
        if (!serialized_) {
            uint32_t n_total = model.n_total_params();
            uint32_t n_arms = this->n_arms();
            s << n_total << n_arms;
            serialized_ = true;
        }

        upper_bd_ /= n_;                
        grad_buff_ /= n_;

        // serialize 1/N sum_{i=1}^N 1_{rej hyp i}
        s << upper_bd_;

        // replace the matrix with upper bound for constant term
        upper_bd_.array() = upper_bd_constant(delta);
        s << upper_bd_;

        // serialize gradient (for all components)
        s << grad_buff_;

        // add upper bound for gradient term and hessian term
        upper_bd_grad_hess(
                mean_idxer_range, delta, grid_radius,
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

        mat_t cache;
        vec_t buff;

        while (us.get()) {

            // read batch of constant matrix
            us >> cache;
            Eigen::Map<mat_t> viewer(
                    c_vec.data() + offset,
                    cache.rows(),
                    cache.cols()
                    );
            viewer = cache;

            // read batch of constant upper bd matrix
            us >> cache;
            new (&viewer) Eigen::Map<mat_t>(
                    c_bd_vec.data() + offset,
                    cache.rows(),
                    cache.cols()
                    );
            viewer = cache;

            // read batch of gradient vector
            us >> buff;
            new (&viewer) Eigen::Map<vec_t>(
                    buff.data(),
                    cache.size(),
                    grad_mat.cols()
                    );
            grad_mat.block(offset, 0, cache.size(), grad_mat.cols())
                = viewer;

            // read batch of gradient/hessian bounds
            Eigen::Map<vec_t> grad_bd_viewer(
                    grad_bd_vec.data() + offset,
                    cache.cols()
                    );
            Eigen::Map<vec_t> hess_bd_viewer(
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
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;

    // This matrix will be used to store the final upper bound.
    // To save memory, it will be used to store the rejection proportions during update.
    mat_t upper_bd_;
    vec_t grad_buff_;
    size_t n_ = 0;              // number of updates
    bool serialized_ = false;   // true iff serialize() has been called.
};


} // namespace kevlar
