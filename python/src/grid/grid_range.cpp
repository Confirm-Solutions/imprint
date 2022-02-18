#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> // must enable for automatic conversion of Eigen
#include <grid/grid.hpp>
#include <kevlar_bits/grid/grid_range.hpp>

namespace kevlar {
namespace grid {

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
        .def(py::pickle(
            [](const gr_t& p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(
                        p.get_thetas(),
                        p.get_radii(), 
                        p.get_sim_sizes());
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 3) {
                    throw std::runtime_error("Invalid state!");
                }

                using t_t = std::decay_t<decltype(std::declval<gr_t>().get_thetas())>;
                using r_t = std::decay_t<decltype(std::declval<gr_t>().get_radii())>;
                using s_t = std::decay_t<decltype(std::declval<gr_t>().get_sim_sizes())>;

                auto thetas = t[0].cast<t_t>();
                auto radii = t[1].cast<r_t>();
                auto sim_sizes = t[2].cast<s_t>();

                /* Create a new C++ instance */
                gr_t p(thetas.rows(), 
                       thetas.cols());
                p.get_thetas() = thetas;
                p.get_radii() = radii;
                p.get_sim_sizes() = sim_sizes;

                return p;
            }))
        ;
}

} // namespace grid
} // namespace kevlar
