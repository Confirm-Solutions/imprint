#pragma once
#include <pybind11/pybind11.h>

#include <kevlar_bits/util/types.hpp>

namespace kevlar {
namespace model {

namespace py = pybind11;

template <class SGSB>
struct PySimGlobalStateBase : SGSB {
    using base_t = SGSB;
    using typename base_t::gen_t;
    using typename base_t::interface_t;
    using typename base_t::uint_t;
    using typename base_t::value_t;

    using base_t::base_t;

    struct SimState;

    using sim_state_t = SimState;

    // See: https://github.com/pybind/pybind11/issues/673#issuecomment-280883981
    std::unique_ptr<typename interface_t::sim_state_t> make_sim_state()
        const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function overload = pybind11::get_overload(
            static_cast<const base_t*>(this), "make_sim_state");
        auto o = overload();
        // Make a new copy of the object.
        // This is fine since on the Python side,
        // when the user calls the Python overloaded version,
        // we intercept the call here.
        auto shptr = pybind11::cast<sim_state_t*>(o);
        return std::make_unique<sim_state_t>(*shptr);
    }
};

template <class SGSB>
struct PySimGlobalStateBase<SGSB>::SimState : base_t::sim_state_t {
    using outer_t = PySimGlobalStateBase;
    using base_t = typename outer_t::base_t::sim_state_t;

    using base_t::base_t;

    void simulate(gen_t& gen,
                  Eigen::Ref<colvec_type<uint_t>> rej_len) override {
        PYBIND11_OVERRIDE_PURE(void, base_t, simulate, gen, rej_len);
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
    using py_sbs_t = PySimGlobalStateBase<sbs_t>;
    py::class_<sbs_t, py_sbs_t>(m, "SimGlobalStateBase")
        .def(py::init<>())
        .def("make_sim_state", &sbs_t::make_sim_state);
    ;

    using ss_t = typename sbs_t::sim_state_t;
    using py_ss_t = typename py_sbs_t::sim_state_t;
    py::class_<ss_t, py_ss_t>(m, "SimStateBase")
        .def(py::init<>())
        .def("simulate", &ss_t::simulate, py::arg("gen"),
             py::arg("rejection_length"))
        .def("score", &ss_t::score, py::arg("gridpt_idx"), py::arg("output"));
}

template <class KBSB>
void add_kevlar_bound_state_base(pybind11::module_& m) {
    using kbs_t = KBSB;
    py::class_<kbs_t>(m, "KevlarBoundStateBase")
        .def("apply_eta_jacobian", &kbs_t::apply_eta_jacobian,
             py::arg("gridpt_idx"), py::arg("v"), py::arg("output"))
        .def("covar_quad", &kbs_t::covar_quadform, py::arg("gridpt_idx"),
             py::arg("v"))
        .def("hessian_quadform_bound", &kbs_t::hessian_quadform_bound,
             py::arg("gridpt_idx"), py::arg("tile_idx"), py::arg("v"))
        .def("n_natural_params", &kbs_t::n_natural_params);
}

}  // namespace model
}  // namespace kevlar
