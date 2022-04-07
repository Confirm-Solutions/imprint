#pragma once
#include <cmath>
#include <kevlar_bits/util/macros.hpp>
#include <limits>

namespace kevlar {
namespace stat {

template <class ZType, class VarType>
KEVLAR_STRONG_INLINE ZType unpaired_z_test_stat(ZType z1, ZType z2, VarType v1,
                                                VarType v2) {
    auto v = v1 + v2;
    auto dz = (z1 - z2);
    return (v <= 0)
               ? std::copysign(1.0, dz) * std::numeric_limits<ZType>::infinity()
               : dz / std::sqrt(v);
}

}  // namespace stat
}  // namespace kevlar
