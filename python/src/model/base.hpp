#pragma once
#include <pybind11/pybind11.h>

namespace kevlar {
namespace model {

namespace py = pybind11;

template <class MB>
void add_model_base(py::module_& m) {
    using mb_t = MB;
    py::class_<mb_t>(m, "ModelBase")
        .def("cov_quad", &mb_t::cov_quad)
        .def("max_cov_quad", &mb_t::max_cov_quad)
        .def("n_models", &mb_t::n_models)
        .def("grid_range", &mb_t::grid_range,
             py::return_value_policy::reference_internal)
        .def("make_state", &mb_t::make_state);
}

template <class MSB>
void add_model_state_base(pybind11::module_& m) {
    using msb_t = MSB;
    py::class_<msb_t>(m, "ModelStateBase")
        .def("gen_rng", &msb_t::gen_rng)
        .def("gen_suff_stat", &msb_t::gen_suff_stat)
        .def("rej_len", &msb_t::rej_len)
        .def("grad", &msb_t::grad)
        .def("grid_range", &msb_t::grid_range,
             py::return_value_policy::reference_internal);
}

template <class CKTB>
void add_control_k_treatment_base(pybind11::module_& m) {
    using cktb_t = CKTB;
    py::class_<cktb_t>(m, "ControlkTreatmentBase")
        .def(py::init<size_t, size_t, size_t>())
        .def("n_arms", &cktb_t::n_arms)
        .def("ph2_size", &cktb_t::ph2_size)
        .def("n_samples", &cktb_t::n_samples);
}

}  // namespace model
}  // namespace kevlar
