#pragma once
#include <pybind11/pybind11.h>

namespace kevlar {
namespace grid {

namespace py = pybind11;

template <class AdaGridInternalType
        , class UpperBoundType
        , class GRType
        , class FNType>
void add_adagrid_internal(py::module_& m) 
{
    using ada_t = AdaGridInternalType;
    using ub_t = UpperBoundType;
    using gr_t = GRType;
    using fn_t = FNType;
    py::class_<ada_t>(m, "AdaGridInternal")
        .def(py::init<>())
        .def("update", 
                &ada_t::template update<
                    ub_t,
                    gr_t&,
                    gr_t&,
                    fn_t>)
        ;
}

} // namespace grid
} // namespace kevlar
