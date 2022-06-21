#pragma once
#include <pybind11/pybind11.h>

#include <imprint_bits/util/types.hpp>

namespace imprint {
namespace model {

namespace py = pybind11;

template <class FSAS>
void add_fixed_single_arm_size(py::module_& m) {
    using base_t = FSAS;
    py::class_<base_t>(m, "FixedSingleArmSize")
        .def("n_arms", &base_t::n_arms)
        .def("n_arm_samples", &base_t::n_arm_samples);
}

}  // namespace model
}  // namespace imprint
