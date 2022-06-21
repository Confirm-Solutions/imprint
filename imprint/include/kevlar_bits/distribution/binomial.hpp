#pragma once
#include <imprint_bits/util/macros.hpp>
#include <imprint_bits/util/math.hpp>
#include <random>

namespace imprint {
namespace distribution {

template <class IntType>
struct Binomial {
    using value_t = IntType;

   private:
    std::binomial_distribution<value_t> binom_dist_;

   public:
    Binomial(value_t n, double p) : binom_dist_(n, p) {}

    /*
     * Samples a single Binomial sample with parameter n, p.
     */
    template <class GenType>
    auto sample(GenType&& gen) {
        return binom_dist_(gen);
    }

    /*
     * Computes the score of a binomial distribution with parameters n, p.
     * The score is given by:
     *      t - n * p
     * where t is the count.
     * The parameters t, n, p are all array-like with the same underlying value
     * type.
     */
    template <class TType, class NType, class PType>
    IMPRINT_STRONG_INLINE static auto score(const TType& t, const NType& n,
                                            const PType& p) {
        return t - n * p;
    }

    /*
     * Computes the quadratic form (of v) of the covariance matrix
     * of the count evaluated at p.
     * If n has length d, the function assumes a random variable of length d
     * with each component i representing the count from
     * sampling n[i] i.i.d. samples of Bernoulli from p[i].
     * This function assumes that the binomial r.v.'s are independent
     * with array-like parameters n, p, v with the same underlying value type.
     */
    template <class NType, class PType, class VType>
    IMPRINT_STRONG_INLINE static auto covar_quadform(const NType& n,
                                                     const PType& p,
                                                     const VType& v) {
        return (n * v.square() * p * (1.0 - p)).sum();
    }

    /*
     * Computes the transformation from natural parameter to mean parameter.
     * nat is an array-like object with each component representing
     * a binomial natural parameter.
     */
    template <class NatType>
    IMPRINT_STRONG_INLINE static auto natural_to_mean(const NatType& nat) {
        return sigmoid(nat);
    }
};

}  // namespace distribution
}  // namespace imprint
