#pragma once
#include <kevlar_bits/util/types.hpp>

namespace kevlar {

template <class ValueType>
auto make_colvec(std::initializer_list<ValueType> l) {
    colvec_type<ValueType> out(l.size());
    auto it = l.begin();
    for (int i = 0; i < out.size(); ++i, ++it) {
        out[i] = (*it);
    }
    return out;
}

}  // namespace kevlar
