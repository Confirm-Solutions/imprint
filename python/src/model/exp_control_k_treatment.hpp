#pragma once
#include <pybind11/pybind11.h>

namespace kevlar {
namespace model {

namespace py = pybind11;

template <class ECKT>
void add_exp_control_k_treatment(py::module_& m)
{
    using eckt_t = ECKT;
    using ckt_t = typename eckt_t::base_t;
    using mb_t = typename eckt_t::model_base_t;
    using value_t = typename eckt_t::value_t;
    py::class_<eckt_t, ckt_t, mb_t>(m, "ExpControlkTreatment")
        .def(py::init<
                size_t,
                value_t,
                const Eigen::Ref<const colvec_type<double>>&
            >())
        .def("set_grid_range", &eckt_t::set_grid_range)
        .def("set_thresholds", &eckt_t::set_thresholds)
        .def(py::pickle(
            [](const eckt_t& p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(
                        p.n_samples(),
                        p.censor_time__(),
                        p.thresholds__(),
                        p.n_gridpts__(),
                        p.buff__()
                        );
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 5) {
                    throw std::runtime_error("Invalid state!");
                }

                /* Create a new C++ instance */
                eckt_t p(t[0].cast<size_t>(),
                         t[1].cast<value_t>(),
                         t[2].cast<Eigen::Ref<const colvec_type<double> >>());
                auto n_gridpts = t[3].cast<
                    std::decay_t<decltype(std::declval<eckt_t>().n_gridpts__())>
                    >();
                auto&& buff = t[4].cast<
                    std::decay_t<decltype(std::declval<eckt_t>().buff__())>
                    >();
                p.set_internal(n_gridpts, buff);
                return p;
            }))
        ;

    using state_t = typename eckt_t::state_t;
    using base_t = typename state_t::model_state_base_t;
    py::class_<state_t, base_t>(m, "ExpControlkTreatmentState")
        .def(py::init<const eckt_t&>())
        ;
}

} // namespace model
} // namespace kevlar
