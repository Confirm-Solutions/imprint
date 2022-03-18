#pragma once
#include <pybind11/pybind11.h>
#include <kevlar_bits/driver/fit.hpp>

namespace kevlar {
namespace driver {

template <class GenType, class ModelBaseType, class GridRangeType,
          class InterSumType>
inline void add_fit(pybind11::module_& m) {
    using gen_t = GenType;
    using model_t = ModelBaseType;
    using gr_t = GridRangeType;
    using is_t = InterSumType;

    m.def("fit", fit<gen_t, model_t, gr_t, is_t>);
}

}  // namespace driver
}  // namespace kevlar
