#include <model/model.hpp>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <kevlar_bits/model/binomial_control_k_treatment.hpp>
#include <kevlar_bits/util/range/grid_range.hpp>
#include <random>
#include <iostream>

namespace kevlar {
namespace model {

namespace py = pybind11;

void add_binomial_control_k_treatment(py::module_& m)
{
    using ckt_t = kevlar::ControlkTreatmentBase;
    using bckt_t = kevlar::BinomialControlkTreatment<double, uint32_t>;
    py::class_<bckt_t, ckt_t>(m, "BinomialControlkTreatment")
        .def(py::init<
                size_t,
                size_t,
                size_t,
                const kevlar::GridRange<double, uint32_t>&,
                double>())
        .def("make_state", &bckt_t::make_state)
        .def("n_gridpts", &bckt_t::n_gridpts)
        .def("tr_cov", &bckt_t::tr_cov)
        .def("tr_max_cov", &bckt_t::tr_max_cov)
        .def(py::pickle(
            [](const bckt_t& p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(
                        p.get_probs_unique(),
                        p.get_strides(),
                        p.get_probs(),
                        p.get_gbits(),
                        p.get_threshold(),
                        p.n_arms(),
                        p.ph2_size(),
                        p.n_samples()
                        );
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 8) {
                    throw std::runtime_error("Invalid state!");
                }

                /* Create a new C++ instance */
                bckt_t p(t[5].cast<size_t>(),
                         t[6].cast<size_t>(),
                         t[7].cast<size_t>());
                p.get_probs_unique() = t[0].cast<
                    std::decay_t<decltype(std::declval<bckt_t>().get_probs_unique())>
                    >();
                p.get_strides() = t[1].cast<
                    std::decay_t<decltype(std::declval<bckt_t>().get_strides())>
                    >();
                p.get_probs() = t[2].cast<
                    std::decay_t<decltype(std::declval<bckt_t>().get_probs())>
                    >();
                p.get_gbits() = t[3].cast<
                    std::decay_t<decltype(std::declval<bckt_t>().get_gbits())>
                    >();
                p.get_threshold() = t[4].cast<
                    std::decay_t<decltype(std::declval<bckt_t>().get_threshold())>
                    >();
                return p;
            }))
        ;

    using base_t = ModelStateBase<double, uint32_t>;
    using state_t = typename internal::traits<bckt_t>::state_t;
    py::class_<state_t, base_t>(m, "BinomialControlkTreatmentState")
        .def(py::init<const bckt_t&>())
        .def("gen_rng", &state_t::gen_rng<std::mt19937&>)
        .def("gen_suff_stat", &state_t::gen_suff_stat)
        .def("n_models", &state_t::n_models)
        .def("n_gridpts", &state_t::n_gridpts)
        .def("n_params", &state_t::n_params)
        ;
}

} // namespace model
} // namespace kevlar
