#pragma once
#include <pybind11/pybind11.h>

namespace kevlar {
namespace std {
namespace random {

namespace py = pybind11;

template <class MT19937>
void add_mt19937(pybind11::module_& m)
{
    using mt19937_t = MT19937;
    py::class_<mt19937_t>(m, "mt19937")
        .def(py::init<>())
        .def("seed", 
                static_cast<void (mt19937_t::*)(typename mt19937_t::result_type)>(&mt19937_t::seed),
                "Reinitializes the internal state of the random-number engine using new seed value.",
                py::arg("seed")=mt19937_t::default_seed)
        .def(py::pickle(
            [](const mt19937_t& p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(p);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 1) {
                    throw ::std::runtime_error("Invalid state!");
                }

                /* Create a new C++ instance */
                mt19937_t p(t[0].cast<mt19937_t>());

                return p;
            }))
        ;
}

} // namespace random
} // namespace std
} // namespace kevlar
