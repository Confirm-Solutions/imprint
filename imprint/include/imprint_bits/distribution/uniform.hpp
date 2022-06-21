#pragma once
#include <cstddef>
#include <imprint_bits/util/macros.hpp>
#include <random>

namespace imprint {
namespace distribution {

template <class ValueType>
struct Uniform {
    using value_t = ValueType;

   private:
    std::uniform_real_distribution<value_t> unif_;

   public:
    Uniform(value_t min, value_t max) : unif_(min, max) {}

    /*
     * Generates i.i.d. uniform samples using the distribution object unif
     * and the RNG gen of shape (m, n), and stores the result in out.
     */
    template <class GenType, class OutType>
    IMPRINT_STRONG_INLINE void sample(size_t m, size_t n, GenType&& gen,
                                      OutType&& out) {
        out = out.NullaryExpr(m, n, [&](auto, auto) { return unif_(gen); });
    }
};

}  // namespace distribution
}  // namespace imprint
