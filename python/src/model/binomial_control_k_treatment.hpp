#pragma once
#include <pybind11/pybind11.h>
#include <kevlar_bits/util/types.hpp>

namespace kevlar {
namespace model {

namespace py = pybind11;

template <class BCKT, class MT19937>
void add_binomial_control_k_treatment(py::module_& m)
{
    using bckt_t = BCKT;
    using ckt_t = typename bckt_t::base_t;
    using mb_t = typename bckt_t::model_base_t;

    using value_t = typename bckt_t::value_t;
    py::class_<bckt_t, ckt_t, mb_t>(m, "BinomialControlkTreatment")
        .def(py::init<
                size_t,
                size_t,
                size_t,
                Eigen::Ref<const colvec_type<value_t> >
            >())
        .def("set_grid_range", &bckt_t::set_grid_range)
        .def("make_state", &bckt_t::make_state)
        .def("n_models", &bckt_t::n_models)
        .def("set_thresholds", &bckt_t::set_thresholds)
        .def(py::pickle(
            [](const bckt_t& p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(
                        p.probs_unique(),
                        p.strides(),
                        p.probs(),
                        p.gbits(),
                        p.thresholds(),
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
                auto thresh = t[4].cast<
                    std::decay_t<decltype(std::declval<bckt_t>().thresholds())>
                    >();
                bckt_t p(t[5].cast<size_t>(),
                         t[6].cast<size_t>(),
                         t[7].cast<size_t>(),
                         thresh);
                auto&& probs_unique = t[0].cast<
                    std::decay_t<decltype(std::declval<bckt_t>().probs_unique())>
                    >();
                auto&& strides = t[1].cast<
                    std::decay_t<decltype(std::declval<bckt_t>().strides())>
                    >();
                auto&& probs = t[2].cast<
                    std::decay_t<decltype(std::declval<bckt_t>().probs())>
                    >();
                auto&& gbits = t[3].cast<
                    std::decay_t<decltype(std::declval<bckt_t>().gbits())>
                    >();
                p.set_internal(probs_unique, strides, probs, gbits);
                return p;
            }))
        ;

    using state_t = typename bckt_t::state_t;
    using base_t = typename state_t::model_state_base_t;
    using mt19937_t = MT19937;
    py::class_<state_t, base_t>(m, "BinomialControlkTreatmentState")
        .def(py::init<const bckt_t&>())
        .def("gen_rng", &state_t::template gen_rng<mt19937_t&>)
        .def("gen_suff_stat", &state_t::gen_suff_stat)
        ;
}

} // namespace model
} // namespace kevlar
