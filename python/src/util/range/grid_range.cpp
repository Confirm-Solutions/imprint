#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> // must enable for automatic conversion of Eigen
#include <kevlar_bits/util/range/grid_range.hpp>

namespace kevlar {
namespace util {
namespace range {

namespace py = pybind11;

void add_grid_range(py::module_& m) 
{
    using gr_t = GridRange<double, uint32_t>;
    py::class_<gr_t>(m, "GridRange")
        .def(py::init<>())
        .def(py::init<uint32_t, uint32_t>())
        .def("size", &gr_t::size)
        .def("dim", &gr_t::dim)
        .def("get_thetas", 
                py::overload_cast<>(&gr_t::get_thetas), 
                py::return_value_policy::reference_internal)
        .def("get_thetas_const", 
                py::overload_cast<>(&gr_t::get_thetas, py::const_),
                py::return_value_policy::reference_internal)
        .def("get_radii",
                py::overload_cast<>(&gr_t::get_radii),
                py::return_value_policy::reference_internal)
        .def("get_radii_const",
                py::overload_cast<>(&gr_t::get_radii, py::const_),
                py::return_value_policy::reference_internal)
        .def("get_sim_sizes",
                py::overload_cast<>(&gr_t::get_sim_sizes),
                py::return_value_policy::reference_internal)
        .def("get_sim_sizes_const",
                py::overload_cast<>(&gr_t::get_sim_sizes, py::const_),
                py::return_value_policy::reference_internal)
        .def("get_sim_sizes_rem",
                py::overload_cast<>(&gr_t::get_sim_sizes_rem),
                py::return_value_policy::reference_internal)
        .def("get_sim_sizes_rem_const",
                py::overload_cast<>(&gr_t::get_sim_sizes_rem, py::const_),
                py::return_value_policy::reference_internal)
        ;
}

} // namespace range
} // namespace util
} // namespace kevlar
