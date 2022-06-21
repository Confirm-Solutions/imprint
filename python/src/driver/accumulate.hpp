#pragma once
#include <pybind11/pybind11.h>

#include <imprint_bits/driver/accumulate.hpp>

namespace imprint {
namespace driver {

namespace py = pybind11;

template <class SGSType, class GridRangeType, class AccumType>
inline void add_accumulate(pybind11::module_& m) {
    using sgs_t = SGSType;
    using ss_t = typename sgs_t::sim_state_t;
    using vec_ss_t = std::vector<ss_t*>;
    using gr_t = GridRangeType;
    using acc_t = AccumType;

    m.def(
        "accumulate",
        [](const vec_ss_t& vec_ss, const gr_t& gr, acc_t& accum,
           size_t sim_size, size_t n_threads) {
            // release GIL before running long C++ function
            py::gil_scoped_release release;
            accumulate_(vec_ss, gr, accum, sim_size, n_threads);
        },
        py::arg("vec_sim_states"), py::arg("grid_range"), py::arg("accum"),
        py::arg("sim_size"), py::arg("n_threads"));
}

}  // namespace driver
}  // namespace imprint
