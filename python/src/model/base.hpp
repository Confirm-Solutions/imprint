#pragma once
#include <pybind11/pybind11.h>

#include <imprint_bits/util/types.hpp>

namespace imprint {
namespace model {

namespace py = pybind11;

template <class SS>
struct PySimStateBase : SS {
    using base_t = SS;
    using typename base_t::uint_t;
    using typename base_t::value_t;

    using base_t::base_t;

    void simulate(Eigen::Ref<colvec_type<uint_t>> rej_len) override {
        PYBIND11_OVERRIDE_PURE(void, base_t, simulate, rej_len);
    }

    void score(size_t gridpt_idx,
               Eigen::Ref<colvec_type<value_t>> out) const override {
        PYBIND11_OVERRIDE_PURE(void, base_t, score, gridpt_idx, out);
    }
};

template <class MB>
void add_model_base(py::module_& m) {
    using mb_t = MB;
    using value_t = typename mb_t::value_t;
    py::class_<mb_t>(m, "ModelBase")
        .def(py::init<>())
        .def(py::init<const Eigen::Ref<const colvec_type<value_t>>&>())
        .def("n_models", &mb_t::n_models)
        .def("critical_values", py::overload_cast<>(&mb_t::critical_values),
             py::return_value_policy::reference_internal)
        .def("critical_values",
             py::overload_cast<>(&mb_t::critical_values, py::const_),
             py::return_value_policy::reference_internal)
        .def("critical_values",
             py::overload_cast<const Eigen::Ref<const colvec_type<value_t>>&>(
                 &mb_t::critical_values),
             py::arg("critical_values"));
}

template <class SGSB>
void add_sim_global_state_base(pybind11::module_& m) {
    using sbs_t = SGSB;
    py::class_<sbs_t>(m, "SimGlobalStateBase")
        .def("make_sim_state", &sbs_t::make_sim_state);
    ;

    using ss_t = typename sbs_t::sim_state_t;
    using py_ss_t = PySimStateBase<ss_t>;
    py::class_<ss_t, py_ss_t>(m, "SimStateBase")
        .def(py::init<>())
        .def("simulate", &ss_t::simulate, py::arg("rejection_length"))
        .def("score", &ss_t::score, py::arg("gridpt_idx"), py::arg("output"));
}

template <class KBSB>
void add_imprint_bound_state_base(pybind11::module_& m) {
    using kbs_t = KBSB;
    py::class_<kbs_t>(m, "ImprintBoundStateBase")
        .def("apply_eta_jacobian", &kbs_t::apply_eta_jacobian,
             py::arg("gridpt_idx"), py::arg("v"), py::arg("output"))
        .def("covar_quad", &kbs_t::covar_quadform, py::arg("gridpt_idx"),
             py::arg("v"))
        .def("hessian_quadform_bound", &kbs_t::hessian_quadform_bound,
             py::arg("gridpt_idx"), py::arg("tile_idx"), py::arg("v"))
        .def("n_natural_params", &kbs_t::n_natural_params);
}

}  // namespace model
}  // namespace imprint
