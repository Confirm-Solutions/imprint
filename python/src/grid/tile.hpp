#pragma once
#include <pybind11/pybind11.h>
#include <bitset>
#include <kevlar_bits/util/types.hpp>

namespace kevlar {
namespace grid {

namespace py = pybind11;

template <class TileType>
void add_tile(py::module_& m) 
{
    using tile_t = TileType;
    using value_t = typename tile_t::value_t;
    py::class_<tile_t>(m, "Tile")
        .def(py::init<
                const Eigen::Ref<const colvec_type<value_t>>,
                const Eigen::Ref<const colvec_type<value_t>> >())
        .def("check_null", &tile_t::check_null)
        .def("set_null", 
                py::overload_cast<size_t, bool>(&tile_t::set_null),
                "Sets the null hypothesis at index with boolean.",
                py::arg("i"),
                py::arg("b")=true)
        .def(py::pickle(
            [](const tile_t& p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(
                        p.bits__().to_ulong(),
                        p.vertices__());
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 2) {
                    throw std::runtime_error("Invalid state!");
                }

                using b_t = std::decay_t<decltype(std::declval<tile_t>().bits__().to_ulong())>;
                using v_t = std::decay_t<decltype(std::declval<tile_t>().vertices__())>;

                std::bitset<tile_t::n_bits> bits = t[0].cast<b_t>();
                auto vertices = t[1].cast<v_t>();

                // NOTE: for now, it's ok to set these as nullptrs.
                // The only time this gets pickled is when GridRange gets pickled.
                Eigen::Map<const colvec_type<value_t>> center(nullptr, 0);
                Eigen::Map<const colvec_type<value_t>> radius(nullptr, 0);

                /* Create a new C++ instance */
                tile_t p(center, radius);
                p.bits__() = bits;
                p.vertices__() = vertices;

                return p;
            }))
        ;
}

} // namespace grid
} // namespace kevlar
