#include <model/model.hpp>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <kevlar_bits/model/binomial_control_k_treatment.hpp>
#include <kevlar_bits/grid/grid_range.hpp>
#include <random>
#include <iostream>

namespace kevlar {
namespace model {

namespace py = pybind11;

void add_binomial_control_k_treatment(py::module_& m)
{
    using mb_t = ModelBase<double>;
    using ckt_t = kevlar::ControlkTreatmentBase;
    using bckt_t = kevlar::BinomialControlkTreatment<double, uint32_t>;
    py::class_<bckt_t, ckt_t, mb_t>(m, "BinomialControlkTreatment")
        .def(py::init<
                size_t,
                size_t,
                size_t,
                Eigen::Ref<const colvec_type<double> >
            >())
        .def("set_grid_range", &bckt_t::set_grid_range<
                const kevlar::GridRange<double, uint32_t>&,
                const std::function<bool(uint32_t, const Eigen::Ref<const colvec_type<double>>&)>&
                >)
        .def("make_state", &bckt_t::make_state)
        .def("n_gridpts", &bckt_t::n_gridpts)
        .def(py::pickle(
            [](const bckt_t& p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(
                        p.get_probs_unique(),
                        p.get_strides(),
                        p.get_probs(),
                        p.get_gbits(),
                        p.get_thresholds(),
                        p.get_null_hypo(),
                        p.n_arms(),
                        p.ph2_size(),
                        p.n_samples()
                        );
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 9) {
                    throw std::runtime_error("Invalid state!");
                }

                /* Create a new C++ instance */
                auto thresh = t[4].cast<
                    std::decay_t<decltype(std::declval<bckt_t>().get_thresholds())>
                    >();
                bckt_t p(t[6].cast<size_t>(),
                         t[7].cast<size_t>(),
                         t[8].cast<size_t>(),
                         thresh);
                auto&& probs_unique = t[0].cast<
                    std::decay_t<decltype(std::declval<bckt_t>().get_probs_unique())>
                    >();
                auto&& strides = t[1].cast<
                    std::decay_t<decltype(std::declval<bckt_t>().get_strides())>
                    >();
                auto&& probs = t[2].cast<
                    std::decay_t<decltype(std::declval<bckt_t>().get_probs())>
                    >();
                auto&& gbits = t[3].cast<
                    std::decay_t<decltype(std::declval<bckt_t>().get_gbits())>
                    >();
                auto&& null_hypo = t[5].cast<
                    std::decay_t<decltype(std::declval<bckt_t>().get_null_hypo())>
                    >();
                p.set_internal(probs_unique, strides, probs, null_hypo, gbits);
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
