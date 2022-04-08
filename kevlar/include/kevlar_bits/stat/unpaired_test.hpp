#pragma once
#include <cmath>
#include <kevlar_bits/util/macros.hpp>
#include <limits>

namespace kevlar {
namespace stat {

template <class ValueType>
struct UnpairedTest {
    using value_t = ValueType;

    template <class IntType, class NType>
    KEVLAR_STRONG_INLINE static value_t binom_stat(IntType x1, IntType x2,
                                                   NType n) {
        IntType dx = x1 - x2;
        auto v = (x1 * (n - x1) + x2 * (n - x2));
        return (v <= 0) ? std::copysign(1, dx) *
                              std::numeric_limits<value_t>::infinity()
                        : (dx * std::sqrt(n)) / std::sqrt(v);
    }

    KEVLAR_STRONG_INLINE
    static value_t z_stat(value_t z1, value_t z2, value_t v1, value_t v2) {
        auto v = v1 + v2;
        auto dz = (z1 - z2);
        return (v <= 0) ? std::copysign(1.0, dz) *
                              std::numeric_limits<value_t>::infinity()
                        : dz / std::sqrt(v);
    }
};

}  // namespace stat
}  // namespace kevlar
