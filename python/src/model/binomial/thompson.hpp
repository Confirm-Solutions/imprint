#pragma once
#include <pybind11/pybind11.h>

#include <imprint_bits/util/types.hpp>

namespace imprint {
namespace model {
namespace binomial {

namespace py = pybind11;

template <class TModel, class GenType, class ValueType, class UIntType,
          class GridRangeType>
void add_thompson(py::module_& m) {
    using model_t = TModel;
    using arm_base_t = typename model_t::arm_base_t;
    using base_t = typename model_t::base_t;
    using model_value_t = typename model_t::value_t;
    using gen_t = GenType;
    using value_t = ValueType;
    using uint_t = UIntType;
    using gr_t = GridRangeType;

    using sgs_t = typename model_t::template sim_global_state_t<gen_t, value_t,
                                                                uint_t, gr_t>;
    using kbs_t = typename model_t::template imprint_bound_state_t<gr_t>;

    py::class_<model_t, arm_base_t, base_t>(m, "Thompson")
        .def(py::init<size_t, value_t, value_t, value_t,
                      const Eigen::Ref<const colvec_type<model_value_t>>&>(),
             py::arg("n_arm_samples"), py::arg("alpha_prior"),
             py::arg("beta_prior"), py::arg("p_thresh"),
             py::arg("critical_values"))
        // Note that exporting critical_values()
        // somehow hides export of base class version.
        // This is actually what we want!
        .def("critical_values",
             (void(model_t::*)(
                 const Eigen::Ref<const colvec_type<model_value_t>>&)) &
                 model_t::critical_values,
             py::arg("critical_values"))
        .def("critical_values",
             static_cast<decltype(std::declval<const model_t&>()
                                      .critical_values()) (model_t::*)() const>(
                 &model_t::critical_values))
        .def("make_sim_global_state",
             static_cast<sgs_t (model_t::*)(const gr_t&) const>(
                 &model_t::template make_sim_global_state<gen_t, value_t,
                                                          uint_t, gr_t>),
             py::arg("grid_range"))
        .def("make_imprint_bound_state",
             static_cast<kbs_t (model_t::*)(const gr_t&) const>(
                 &model_t::template make_imprint_bound_state<gr_t>))
        .def(py::pickle(
            [](const model_t& p) {  // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(p.n_arm_samples(), p.alpha_prior(),
                                      p.beta_prior(), p.p_threshold(),
                                      p.critical_values());
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 5) {
                    throw std::runtime_error("Invalid state!");
                }

                /* Create a new C++ instance */
                auto&& thresh = t[4].cast<std::decay_t<
                    decltype(std::declval<model_t>().critical_values())>>();
                model_t p(t[0].cast<size_t>(), t[1].cast<value_t>(),
                          t[2].cast<value_t>(), t[3].cast<value_t>(), thresh);
                return p;
            }));

    using sgs_base_t = typename sgs_t::interface_t;
    py::class_<sgs_t, sgs_base_t>(m, "ThompsonSimGlobalState");

    using ss_t = typename sgs_t::sim_state_t;
    using ss_base_t = typename ss_t::base_t;
    py::class_<ss_t, ss_base_t>(m, "ThompsonSimState");
}

}  // namespace binomial
}  // namespace model
}  // namespace imprint
