#pragma once
#include <pybind11/pybind11.h>

namespace imprint {
namespace bound {

namespace py = pybind11;

template <class GRType, class KBStateType, class AccType, class KBType>
void add_typeI_error_bound(py::module_& m) {
    using gr_t = GRType;
    using kbs_t = KBStateType;
    using acc_t = AccType;
    using kb_t = KBType;
    py::class_<kb_t>(m, "TypeIErrorBound")
        .def(py::init<>())
        .def("create", &kb_t::template create<kbs_t&, acc_t, gr_t>,
             "Create and store the components of upper bound.", py::arg("kbs"),
             py::arg("accum"), py::arg("grid_range"), py::arg("delta"),
             py::arg("delta_prop_0to1") = 0.5, py::arg("verbose") = false)
        .def("get", py::overload_cast<>(&kb_t::get, py::const_),
             py::return_value_policy::reference_internal)
        .def("delta_0", py::overload_cast<>(&kb_t::delta_0, py::const_),
             py::return_value_policy::reference_internal)
        .def("delta_0_u", py::overload_cast<>(&kb_t::delta_0_u, py::const_),
             py::return_value_policy::reference_internal)
        .def("delta_1", py::overload_cast<>(&kb_t::delta_1, py::const_),
             py::return_value_policy::reference_internal)
        .def("delta_1_u", py::overload_cast<>(&kb_t::delta_1_u, py::const_),
             py::return_value_policy::reference_internal)
        .def("delta_2_u", py::overload_cast<>(&kb_t::delta_2_u, py::const_),
             py::return_value_policy::reference_internal);
}

}  // namespace bound
}  // namespace imprint
