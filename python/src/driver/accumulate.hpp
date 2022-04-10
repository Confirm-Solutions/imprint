#pragma once
#include <pybind11/pybind11.h>

#include <kevlar_bits/driver/accumulate.hpp>

namespace kevlar {
namespace driver {

template <class SGSType, class GridRangeType, class InterSumType>
inline void add_accumulate(pybind11::module_& m) {
    using sgs_t = SGSType;
    using gr_t = GridRangeType;
    using is_t = InterSumType;

    m.def("accumulate", accumulate<sgs_t, gr_t, is_t>);
}

}  // namespace driver
}  // namespace kevlar
