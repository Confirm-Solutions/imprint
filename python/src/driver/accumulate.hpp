#pragma once
#include <pybind11/pybind11.h>

#include <kevlar_bits/driver/accumulate.hpp>

namespace kevlar {
namespace driver {

namespace py = pybind11;

template <class SGSType, class GridRangeType, class AccumType>
inline void add_accumulate(pybind11::module_& m) {
    using sgs_t = SGSType;
    using gr_t = GridRangeType;
    using acc_t = AccumType;

    m.def("accumulate", accumulate<sgs_t, gr_t, acc_t>,
          py::arg("sim_global_state"), py::arg("grid_range"), py::arg("accum"),
          py::arg("sim_size"), py::arg("seed"), py::arg("n_threads"));
}

}  // namespace driver
}  // namespace kevlar
