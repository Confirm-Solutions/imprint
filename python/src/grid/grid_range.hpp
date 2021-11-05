#pragma once
#include <pybind11/pybind11.h>

namespace kevlar {
namespace grid {

namespace py = pybind11;

template <class GRType, class VecSurfType>
void add_grid_range(py::module_& m) {
    using gr_t = GRType;
    using vec_surf_t = VecSurfType;
    using uint_t = typename gr_t::uint_t;
    py::class_<gr_t>(m, "GridRange")
        .def(py::init<>())
        .def(py::init<uint_t, uint_t>())
        .def("create_tiles", &gr_t::template create_tiles<vec_surf_t>)
        .def("prune", &gr_t::prune)
        .def("n_tiles", py::overload_cast<size_t>(&gr_t::n_tiles, py::const_))
        .def("n_tiles", py::overload_cast<>(&gr_t::n_tiles, py::const_))
        .def("n_gridpts", &gr_t::n_gridpts)
        .def("n_params", &gr_t::n_params)
        .def("thetas", py::overload_cast<>(&gr_t::thetas),
             py::return_value_policy::reference_internal)
        .def("thetas_const", py::overload_cast<>(&gr_t::thetas, py::const_),
             py::return_value_policy::reference_internal)
        .def("radii", py::overload_cast<>(&gr_t::radii),
             py::return_value_policy::reference_internal)
        .def("radii_const", py::overload_cast<>(&gr_t::radii, py::const_),
             py::return_value_policy::reference_internal)
        .def("sim_sizes", py::overload_cast<>(&gr_t::sim_sizes),
             py::return_value_policy::reference_internal)
        .def("sim_sizes_const",
             py::overload_cast<>(&gr_t::sim_sizes, py::const_),
             py::return_value_policy::reference_internal)
        .def("check_null", &gr_t::check_null)
        .def(py::pickle(
            [](const gr_t& p) {  // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(p.thetas(), p.radii(), p.sim_sizes(),
                                      p.n_tiles__(), p.tiles(), p.bits__());
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 6) {
                    throw std::runtime_error("Invalid state!");
                }

                using t_t =
                    std::decay_t<decltype(std::declval<gr_t>().thetas())>;
                using r_t =
                    std::decay_t<decltype(std::declval<gr_t>().radii())>;
                using s_t =
                    std::decay_t<decltype(std::declval<gr_t>().sim_sizes())>;
                using nt_t =
                    std::decay_t<decltype(std::declval<gr_t>().n_tiles__())>;
                using tt_t =
                    std::decay_t<decltype(std::declval<gr_t>().tiles())>;
                using b_t =
                    std::decay_t<decltype(std::declval<gr_t>().bits__())>;

                auto thetas = t[0].cast<t_t>();
                auto radii = t[1].cast<r_t>();
                auto sim_sizes = t[2].cast<s_t>();
                auto n_tiles = t[3].cast<nt_t>();
                auto tiles = t[4].cast<tt_t>();
                auto bits = t[5].cast<b_t>();

                /* Create a new C++ instance */
                gr_t p;
                p.thetas() = thetas;
                p.radii() = radii;
                p.sim_sizes() = sim_sizes;
                p.n_tiles__() = n_tiles;
                p.tiles__() = tiles;
                p.bits__() = bits;
                // std::swap(p.thetas(), thetas);
                // std::swap(p.radii(), radii);
                // std::swap(p.sim_sizes(), sim_sizes);
                // std::swap(p.n_tiles__(), n_tiles);
                // std::swap(p.tiles__(), tiles);
                // std::swap(p.bits__(), bits);

                return p;
            }));
}

}  // namespace grid
}  // namespace kevlar
