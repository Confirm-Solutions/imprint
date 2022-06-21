#pragma once
#include <pybind11/pybind11.h>

namespace imprint {
namespace grid {

namespace py = pybind11;

template <class AdaGridInternalType, class ImprintBoundType, class GRType,
          class ValueType>
void add_adagrid_internal(py::module_& m) {
    using ada_t = AdaGridInternalType;
    using ub_t = ImprintBoundType;
    using gr_t = GRType;
    using value_t = ValueType;
    py::class_<ada_t>(m, "AdaGridInternal")
        .def(py::init<>())
        .def("update", &ada_t::template update<ub_t, gr_t, value_t>);
}

}  // namespace grid
}  // namespace imprint
