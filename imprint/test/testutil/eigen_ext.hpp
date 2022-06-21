#pragma once
#include <imprint_bits/util/types.hpp>

namespace imprint {

template <class ValueType>
auto make_colvec(std::initializer_list<ValueType> l) {
    colvec_type<ValueType> out(l.size());
    auto it = l.begin();
    for (int i = 0; i < out.size(); ++i, ++it) {
        out[i] = (*it);
    }
    return out;
}

}  // namespace imprint
