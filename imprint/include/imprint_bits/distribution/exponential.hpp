#pragma once
#include <cstddef>
#include <imprint_bits/util/macros.hpp>
#include <random>

namespace imprint {
namespace distribution {

template <class ValueType>
struct Exponential {
    using value_t = ValueType;

   private:
    std::exponential_distribution<value_t> exp_;

   public:
    Exponential(value_t scale) : exp_(scale) {}

    /*
     * Generates i.i.d. exponential samples using
     * the RNG gen of shape (m, n), and stores the result in out.
     */
    template <class GenType, class OutType>
    IMPRINT_STRONG_INLINE void sample(size_t m, size_t n, GenType&& gen,
                                      OutType&& out) {
        out = out.NullaryExpr(m, n, [&](auto, auto) { return exp_(gen); });
    }

    /*
     * Computes the quadratic form (of v) of the covariance matrix
     * of the sufficient statistic evaluated at lmda.
     * If n has length d, the function assumes a random variable of length d
     * with each component i representing the sufficient statistic of
     * sampling n[i] i.i.d. samples from lmda[i].
     * It assumes array-like parameters n, lmda, v.
     */
    template <class NType, class LmdaType, class VType>
    IMPRINT_STRONG_INLINE static auto covar_quadform(const NType& n,
                                                     const LmdaType& lmda,
                                                     const VType& v) {
        return (n * (v.square() / lmda.square())).sum();
    }

    /*
     * Computes the score of exponential distribution for n[i] i.i.d.
     * draws of Exp(lmda[i]) with sufficient statistic t[i].
     * Returns an array-like expression with the score for each i.
     */
    template <class TType, class NType, class LmdaType>
    IMPRINT_STRONG_INLINE static auto score(const TType& t, const NType& n,
                                            const LmdaType& lmda) {
        return t - n * (1 / lmda);
    }

    /*
     * Computes the transformation from natural parameter to mean parameter.
     * nat is an array-like object with each component representing
     * an exponential natural parameter.
     */
    template <class NatType>
    IMPRINT_STRONG_INLINE static auto natural_to_mean(const NatType& nat) {
        return -nat;
    }
};

}  // namespace distribution
}  // namespace imprint
