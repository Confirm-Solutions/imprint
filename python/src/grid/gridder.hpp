#pragma once
#include <pybind11/pybind11.h>

#include <imprint_bits/util/types.hpp>

namespace imprint {
namespace grid {

namespace py = pybind11;

template <class GridderType, class ValueType, class UIntType>
colvec_type<ValueType> make_grid_wrap(UIntType n, ValueType l, ValueType u) {
    return GridderType::make_grid(n, l, u);
}

template <class GridderType, class ValueType, class UIntType>
void add_gridder(pybind11::module_& m) {
    using gridder_t = GridderType;
    using value_t = ValueType;
    using uint_t = UIntType;
    py::class_<gridder_t>(m, "Gridder")
        .def(py::init<>())
        .def("radius", &gridder_t::template radius<value_t>)
        .def("make_grid", &make_grid_wrap<gridder_t, value_t, uint_t>);
}

}  // namespace grid
}  // namespace imprint
