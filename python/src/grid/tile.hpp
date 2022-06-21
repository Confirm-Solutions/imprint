#pragma once
#include <pybind11/pybind11.h>

#include <imprint_bits/util/types.hpp>

namespace imprint {
namespace grid {

namespace py = pybind11;

template <class TileType>
void add_tile(py::module_& m) {
    using tile_t = TileType;
    using value_t = typename tile_t::value_t;
    py::class_<tile_t>(m, "Tile")
        .def(py::init<const Eigen::Ref<const colvec_type<value_t>>,
                      const Eigen::Ref<const colvec_type<value_t>>>(),
             py::arg("center"),
             py::arg("radius"))
        .def(py::pickle(
            [](const tile_t& p) {  // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(p.vertices__());
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 1) {
                    throw std::runtime_error("Invalid state!");
                }

                using v_t =
                    std::decay_t<decltype(std::declval<tile_t>().vertices__())>;

                auto&& vertices = t[0].cast<v_t>();

                // NOTE: for now, it's ok to set these as nullptrs.
                // The only time this gets pickled is when GridRange gets
                // pickled.
                Eigen::Map<const colvec_type<value_t>> center(nullptr, 0);
                Eigen::Map<const colvec_type<value_t>> radius(nullptr, 0);

                /* Create a new C++ instance */
                tile_t p(center, radius);
                p.vertices__() = vertices;

                return p;
            }));
}

}  // namespace grid
}  // namespace imprint
