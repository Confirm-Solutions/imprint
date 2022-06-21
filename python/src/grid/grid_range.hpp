#pragma once
#include <pybind11/pybind11.h>

#include <imprint_bits/util/types.hpp>

namespace imprint {
namespace grid {

namespace py = pybind11;

template <class GRType, class VecSurfType>
void add_grid_range(py::module_& m) {
    using gr_t = GRType;
    using value_t = typename gr_t::value_t;
    using vec_surf_t = VecSurfType;
    using uint_t = typename gr_t::uint_t;
    py::class_<gr_t>(m, "GridRange")
        .def(py::init<>())
        .def(py::init<uint_t, uint_t>(), py::arg("n_params"),
             py::arg("n_gridpts"))
        .def(py::init<const Eigen::Ref<const mat_type<value_t>>&,
                      const Eigen::Ref<const mat_type<value_t>>&,
                      const Eigen::Ref<const colvec_type<uint_t>>&>(),
             py::arg("thetas"), py::arg("radii"), py::arg("sim_sizes"))
        .def(py::init<const Eigen::Ref<const mat_type<value_t>>&,
                      const Eigen::Ref<const mat_type<value_t>>&,
                      const Eigen::Ref<const colvec_type<uint_t>>&,
                      const vec_surf_t&, bool>(),
             py::arg("thetas"), py::arg("radii"), py::arg("sim_sizes"),
             py::arg("surfaces"), py::arg("prune") = true)
        .def("create_tiles", &gr_t::template create_tiles<vec_surf_t>,
             py::arg("surfaces"))
        .def("prune", &gr_t::prune)
        .def("n_tiles", py::overload_cast<size_t>(&gr_t::n_tiles, py::const_),
             py::arg("gridpt_idx"))
        .def("n_tiles", py::overload_cast<>(&gr_t::n_tiles, py::const_))
        .def("cum_n_tiles", &gr_t::cum_n_tiles)
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
        .def("corners",
             [](gr_t& gr,
                Eigen::Ref<mat_type<value_t, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>& out) {
                 // out is expected to be full of nans and to have shape:
                 // (n_tiles * max_corners, dim).
                 int dim = gr.thetas().rows();
                 colvec_type<value_t> bits(dim);
                 int two_to_dim = std::pow(2, dim);
                 int max_corners = 2 * two_to_dim;

                 // loop over each tile and assign the corners for that tile to
                 // the out array.
                 for (size_t i = 0; i < gr.n_tiles(); i++) {
                     auto& t = gr.tiles__()[i];
                     if (t.is_regular()) {
                         for (int v_idx = 0; v_idx < two_to_dim; v_idx++) {
                             for (int k = 0; k < dim; k++) {
                                 bits(k) =
                                     2 * static_cast<int>(static_cast<bool>(
                                             v_idx & (1 << (dim - 1 - k)))) -
                                     1;
                             }
                             out.row(i * max_corners + v_idx) =
                                 t.regular_vertex(bits);
                         }
                     } else {
                         auto begin = t.begin();
                         auto end = t.end();
                         int v_idx = 0;
                         for (; begin != end; ++begin, v_idx++) {
                             out.row(i * max_corners + v_idx) = *begin;
                         }
                     }
                 }
             })
        .def("check_null",
             py::overload_cast<size_t, size_t>(&gr_t::check_null, py::const_),
             py::arg("tile_idx"), py::arg("hypo_idx"))
        .def("check_null",
             py::overload_cast<size_t, size_t, size_t>(&gr_t::check_null,
                                                       py::const_),
             py::arg("gridpt_idx"), py::arg("rel_tile_idx"),
             py::arg("hypo_idx"))
        .def(py::pickle(
            [](const gr_t& p) {  // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(p.thetas(), p.radii(), p.sim_sizes(),
                                      p.cum_n_tiles__(), p.tiles(), p.bits__());
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
                using nt_t = std::decay_t<
                    decltype(std::declval<gr_t>().cum_n_tiles__())>;
                using tt_t =
                    std::decay_t<decltype(std::declval<gr_t>().tiles())>;
                using b_t =
                    std::decay_t<decltype(std::declval<gr_t>().bits__())>;

                auto thetas = t[0].cast<t_t>();
                auto radii = t[1].cast<r_t>();
                auto sim_sizes = t[2].cast<s_t>();
                auto cum_n_tiles = t[3].cast<nt_t>();
                auto tiles = t[4].cast<tt_t>();
                auto bits = t[5].cast<b_t>();

                /* Create a new C++ instance */
                gr_t p;
                p.thetas() = thetas;
                p.radii() = radii;
                p.sim_sizes() = sim_sizes;
                p.cum_n_tiles__() = cum_n_tiles;
                p.tiles__() = tiles;
                p.bits__() = bits;

                return p;
            }));
}

}  // namespace grid
}  // namespace imprint
