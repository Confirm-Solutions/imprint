#pragma once
#include <cmath>
#include <imprint_bits/util/macros.hpp>
#include <limits>

namespace imprint {
namespace stat {

template <class ValueType>
struct UnpairedTest {
    using value_t = ValueType;

    /*
     * Unpaired test for binomial data from 2 groups with
     * binomial statistics x1, x2. This assumes that both groups
     * have the same size n, which allows for a more optimized computation.
     * It computes:
     *
     *      \frac{x_1 - x_2}{\sqrt{(x_1(n-x_1) + x_2(n-x_2)) / n}}
     *
     * If the variance is non-positive, it is set to $\pm \infty$ depending
     * on the sign of $x_1-x_2$.
     * Note that x1, x2 must be signed integer types.
     */
    template <class IntType, class NType>
    IMPRINT_STRONG_INLINE static value_t binom_stat(IntType x1, IntType x2,
                                                    NType n) {
        IntType dx = x1 - x2;
        auto v = (x1 * (n - x1) + x2 * (n - x2));
        return (v <= 0) ? std::copysign(1, dx) *
                              std::numeric_limits<value_t>::infinity()
                        : (dx * std::sqrt(n)) / std::sqrt(v);
    }

    /*
     * Unpaired z-test for a general two group comparison.
     * This test assumes z1, z2 are the normal values from the two groups
     * with variance v1, v2. It computes
     *
     *      \frac{z_1-z_2}{\sqrt{v_1+v_2}}
     *
     * If the variance is non-positive, it is set to $\pm \infty$ depending
     * on the sign of $z_1-z_2$.
     */
    IMPRINT_STRONG_INLINE
    static value_t z_stat(value_t z1, value_t z2, value_t v1, value_t v2) {
        auto v = v1 + v2;
        auto dz = (z1 - z2);
        return (v <= 0) ? std::copysign(1.0, dz) *
                              std::numeric_limits<value_t>::infinity()
                        : dz / std::sqrt(v);
    }
};

}  // namespace stat
}  // namespace imprint
