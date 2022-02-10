#include <std/random/random.hpp>
#include <random>
#include <cstdint>

namespace kevlar {
namespace std {
namespace random {

namespace py = pybind11;

void add_mt19937(pybind11::module_& m)
{
    using mt19937_t = ::std::mt19937;
    py::class_<mt19937_t>(m, "mt19937")
        .def(py::init<>())
        .def("seed", 
                static_cast<void (mt19937_t::*)(typename mt19937_t::result_type)>(&mt19937_t::seed),
                "Reinitializes the internal state of the random-number engine using new seed value.",
                py::arg("seed")=mt19937_t::default_seed)
        ;
}

} // namespace random
} // namespace std
} // namespace kevlar
