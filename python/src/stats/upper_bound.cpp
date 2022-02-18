#include <pybind11/eigen.h>
#include <stats/stats.hpp>
#include <kevlar_bits/stats/upper_bound.hpp>
#include <kevlar_bits/stats/inter_sum.hpp>
#include <kevlar_bits/grid/grid_range.hpp>
#include <kevlar_bits/model/base.hpp>

namespace kevlar {
namespace stats {

namespace py = pybind11;

void add_upper_bound(py::module_& m)
{
    using model_base_t = ModelBase<double>;
    using is_t = InterSum<double, uint32_t>;
    using grid_range_t = GridRange<double, uint32_t>;
    using ub_t = UpperBound<double>;
    py::class_<ub_t>(m, "UpperBound")
        .def(py::init<>())
        .def("create", &ub_t::create<
                model_base_t,
                is_t,
                grid_range_t>,
                "Create and store the components of upper bound.",
                py::arg("model"), 
                py::arg("inter_sum"),
                py::arg("grid_range"),
                py::arg("delta"),
                py::arg("delta_prop_0to1")=0.5)
        .def("get", &ub_t::get)
        .def("get_delta_0", 
                py::overload_cast<>(&ub_t::get_delta_0),
                py::return_value_policy::reference_internal)
        .def("get_delta_0_const", 
                py::overload_cast<>(&ub_t::get_delta_0, py::const_),
                py::return_value_policy::reference_internal)
        .def("get_delta_0_u", 
                py::overload_cast<>(&ub_t::get_delta_0_u),
                py::return_value_policy::reference_internal)
        .def("get_delta_0_u_const", 
                py::overload_cast<>(&ub_t::get_delta_0_u, py::const_),
                py::return_value_policy::reference_internal)
        .def("get_delta_1", 
                py::overload_cast<>(&ub_t::get_delta_1),
                py::return_value_policy::reference_internal)
        .def("get_delta_1_const", 
                py::overload_cast<>(&ub_t::get_delta_1, py::const_),
                py::return_value_policy::reference_internal)
        .def("get_delta_1_u", 
                py::overload_cast<>(&ub_t::get_delta_1_u),
                py::return_value_policy::reference_internal)
        .def("get_delta_1_u_const", 
                py::overload_cast<>(&ub_t::get_delta_1_u, py::const_),
                py::return_value_policy::reference_internal)
        .def("get_delta_2_u", 
                py::overload_cast<>(&ub_t::get_delta_2_u),
                py::return_value_policy::reference_internal)
        .def("get_delta_2_u_const", 
                py::overload_cast<>(&ub_t::get_delta_2_u, py::const_),
                py::return_value_policy::reference_internal)
        ;
}

} // namespace stats
} // namespace kevlar
