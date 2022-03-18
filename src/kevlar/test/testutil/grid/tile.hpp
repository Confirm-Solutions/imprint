#pragma once
#include <algorithm>
#include <iterator>
#include <kevlar_bits/grid/decl.hpp>

namespace kevlar {

template <class V, size_t N>
inline constexpr bool operator==(const Tile<V, N>& t1, const Tile<V, N>& t2) {
    // check center and radius
    if ((t1.center().array() != t2.center().array()).any()) return false;
    if ((t1.radius().array() != t2.radius().array()).any()) return false;

    // check set of vertices
    auto it1 = t1.begin();
    auto ed1 = t1.end();
    auto it2 = t2.begin();
    auto ed2 = t2.end();

    if (std::distance(it1, ed1) != std::distance(it2, ed2)) return false;
    for (; it1 != ed1; ++it1) {
        if (std::find_if(it2, ed2, [&](const auto& x) {
                return (it1->array() == x.array()).all();
            }) == ed2)
            return false;
    }

    return true;
}

}  // namespace kevlar
