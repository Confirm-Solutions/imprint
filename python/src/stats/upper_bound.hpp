#pragma once
#include <pybind11/pybind11.h>

namespace kevlar {
namespace stats {

namespace py = pybind11;

template <class GRType, class ModelBaseType, class ISType, class UBType>
void add_upper_bound(py::module_& m) {
    using gr_t = GRType;
    using model_base_t = ModelBaseType;
    using is_t = ISType;
    using ub_t = UBType;
    py::class_<ub_t>(m, "UpperBound")
        .def(py::init<>())
        .def("create", &ub_t::template create<model_base_t, is_t, gr_t>,
             "Create and store the components of upper bound.",
             py::arg("model"), py::arg("inter_sum"), py::arg("grid_range"),
             py::arg("delta"), py::arg("delta_prop_0to1") = 0.5,
             py::arg("verbose") = false)
        .def("get", py::overload_cast<>(&ub_t::get, py::const_),
             py::return_value_policy::reference_internal)
        .def("delta_0", py::overload_cast<>(&ub_t::delta_0),
             py::return_value_policy::reference_internal)
        .def("delta_0_const", py::overload_cast<>(&ub_t::delta_0, py::const_),
             py::return_value_policy::reference_internal)
        .def("delta_0_u", py::overload_cast<>(&ub_t::delta_0_u),
             py::return_value_policy::reference_internal)
        .def("delta_0_u_const",
             py::overload_cast<>(&ub_t::delta_0_u, py::const_),
             py::return_value_policy::reference_internal)
        .def("delta_1", py::overload_cast<>(&ub_t::delta_1),
             py::return_value_policy::reference_internal)
        .def("delta_1_const", py::overload_cast<>(&ub_t::delta_1, py::const_),
             py::return_value_policy::reference_internal)
        .def("delta_1_u", py::overload_cast<>(&ub_t::delta_1_u),
             py::return_value_policy::reference_internal)
        .def("delta_1_u_const",
             py::overload_cast<>(&ub_t::delta_1_u, py::const_),
             py::return_value_policy::reference_internal)
        .def("delta_2_u", py::overload_cast<>(&ub_t::delta_2_u),
             py::return_value_policy::reference_internal)
        .def("delta_2_u_const",
             py::overload_cast<>(&ub_t::delta_2_u, py::const_),
             py::return_value_policy::reference_internal);
}

}  // namespace stats
}  // namespace kevlar
