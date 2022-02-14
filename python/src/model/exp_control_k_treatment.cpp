#include <model/model.hpp>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <kevlar_bits/model/exp_control_k_treatment.hpp>
#include <kevlar_bits/util/range/grid_range.hpp>
#include <random>
#include <iostream>

namespace kevlar {
namespace model {

namespace py = pybind11;

void add_exp_control_k_treatment(py::module_& m)
{
    using ckt_t = kevlar::ControlkTreatmentBase;
    using eckt_t = kevlar::ExpControlkTreatment<double, uint32_t>;
    py::class_<eckt_t, ckt_t>(m, "ExpControlkTreatment")
        .def(py::init<
                size_t,
                double,
                double
            >())
        .def("set_grid_range", &eckt_t::set_grid_range<
                const kevlar::GridRange<double, uint32_t>&,
                const std::function<bool(const Eigen::Ref<const colvec_type<double>>&)>&
                >)
        .def("make_state", &eckt_t::make_state)
        .def("n_gridpts", &eckt_t::n_gridpts)
        .def("tr_cov", &eckt_t::tr_cov)
        .def("tr_max_cov", &eckt_t::tr_max_cov)
        .def(py::pickle(
            [](const eckt_t& p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(
                        p.n_samples(),
                        p.get_censor_time(),
                        p.get_threshold(),
                        p.get_n_gridpts(),
                        p.get_buff(),
                        p.get_null_hypo()
                        );
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 6) {
                    throw std::runtime_error("Invalid state!");
                }

                /* Create a new C++ instance */
                eckt_t p(t[0].cast<size_t>(),
                         t[1].cast<double>(),
                         t[2].cast<double>());
                auto n_gridpts = t[3].cast<
                    std::decay_t<decltype(std::declval<eckt_t>().get_n_gridpts())>
                    >();
                auto&& buff = t[4].cast<
                    std::decay_t<decltype(std::declval<eckt_t>().get_buff())>
                    >();
                auto&& null_hypo = t[5].cast<
                    std::decay_t<decltype(std::declval<eckt_t>().get_null_hypo())>
                    >();
                p.set_internal(n_gridpts, buff, null_hypo);
                return p;
            }))
        ;

    using base_t = ModelStateBase<double, uint32_t>;
    using state_t = typename internal::traits<eckt_t>::state_t;
    py::class_<state_t, base_t>(m, "ExpControlkTreatmentState")
        .def(py::init<const eckt_t&>())
        .def("gen_rng", &state_t::gen_rng<std::mt19937&>)
        .def("gen_suff_stat", &state_t::gen_suff_stat)
        .def("n_models", &state_t::n_models)
        .def("n_gridpts", &state_t::n_gridpts)
        .def("n_params", &state_t::n_params)
        ;
}

} // namespace model
} // namespace kevlar
