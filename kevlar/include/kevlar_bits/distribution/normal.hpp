#pragma once
#include <kevlar_bits/util/macros.hpp>
#include <random>

namespace kevlar {
namespace distribution {

template <class ValueType>
struct Normal {
    using value_t = ValueType;

   private:
    std::normal_distribution<value_t> normal_dist_;

   public:
    Normal(value_t loc, value_t scale) : normal_dist_(loc, scale) {}

    /*
     * Samples a single univariate normal random variable
     * given an RNG gen.
     */
    template <class GenType>
    KEVLAR_STRONG_INLINE value_t sample(GenType&& gen) {
        return normal_dist_(gen);
    }
};

}  // namespace distribution
}  // namespace kevlar
