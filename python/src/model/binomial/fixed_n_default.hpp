#pragma once
#include <pybind11/pybind11.h>

#include <kevlar_bits/util/types.hpp>

namespace kevlar {
namespace model {
namespace binomial {

namespace py = pybind11;

template <class SGS, class KBS>
void add_fixed_n_default(py::module_& m) {
    using sgs_t = SGS;
    using sgs_base_t = typename sgs_t::base_t;
    py::class_<sgs_t, sgs_base_t>(m, "SimGlobalStateFixedNDefault");

    using ss_t = typename sgs_t::sim_state_t;
    using ss_base_t = typename ss_t::base_t;
    py::class_<ss_t, ss_base_t>(m, "SimStateFixedNDefault");

    using kbs_t = KBS;
    using kbs_base_t = typename kbs_t::base_t;
    py::class_<kbs_t, kbs_base_t>(m, "KevlarBoundStateFixedNDefault");
}

}  // namespace binomial
}  // namespace model
}  // namespace kevlar
