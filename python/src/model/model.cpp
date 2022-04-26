#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <export_utils/types.hpp>
#include <kevlar_bits/grid/grid_range.hpp>
#include <kevlar_bits/grid/tile.hpp>
#include <kevlar_bits/model/base.hpp>
#include <kevlar_bits/model/binomial/berry_inla.hpp>
#include <kevlar_bits/model/binomial/simple_selection.hpp>
#include <kevlar_bits/model/binomial/thompson.hpp>
#include <kevlar_bits/model/exponential/simple_log_rank.hpp>
#include <model/base.hpp>
#include <model/binomial/fixed_n_default.hpp>
#include <model/binomial/simple_selection.hpp>
#include <model/binomial/thompson.hpp>
#include <model/exponential/fixed_n_log_hazard_rate.hpp>
#include <model/exponential/simple_log_rank.hpp>
#include <model/fixed_single_arm_size.hpp>
#include <model/model.hpp>

namespace kevlar {
namespace model {

namespace py = pybind11;

using value_t = py_double_t;
using uint_t = py_uint_t;
using gen_t = std::mt19937;
using tile_t = grid::Tile<value_t>;
using gr_t = grid::GridRange<value_t, uint_t, tile_t>;

template <int ARMS>
void add_berry_inla_to_module(py::module_& m, const char* model_name,
                              const char* sgs_name, const char* ss_name) {
    using model_t = binomial::BerryINLA<double, ARMS>;

    using sgs_t = typename model_t::template sim_global_state_t<gen_t, value_t,
                                                                uint_t, gr_t>;
    py::class_<model_t, FixedSingleArmSize, ModelBase<double>>(m, model_name)
        .def(py::init<int, const Eigen::Ref<const colvec_type<double>>&,
                      const Eigen::Ref<const colvec_type<double>>&,
                      const Eigen::Ref<const colvec_type<double>>&,
                      const Eigen::Ref<const mat_type<double>>&,
                      const Eigen::Ref<const mat_type<double>>&,
                      const Eigen::Ref<const colvec_type<double>>&,
                      const Eigen::Ref<const colvec_type<double>>&, double,
                      double>(),
             py::arg("arm_size"), py::arg("critical_values"),
             py::arg("efficacy_thresholds"), py::arg("quad_wts"),
             py::arg("cov"), py::arg("neg_precQ"), py::arg("logprecQdet"),
             py::arg("logprior"), py::arg("opt_tol"), py::arg("logit_p1"))
        .def("get_posterior_exceedance_probs",
             &model_t::get_posterior_exceedance_probs)
        .def("make_sim_global_state",
             static_cast<sgs_t (model_t::*)(const gr_t&) const>(
                 &model_t::template make_sim_global_state<gen_t, value_t,
                                                          uint_t, gr_t>),
             py::arg("grid_range"));
    using sgs_base_t = typename sgs_t::interface_t;
    py::class_<sgs_t, sgs_base_t>(m, sgs_name);

    using ss_t = typename sgs_t::sim_state_t;
    using ss_base_t = typename ss_t::base_t;
    py::class_<ss_t, ss_base_t>(m, ss_name);
}
/*
 * Adds binomial models.
 */
void add_binomial_to_module(py::module_& m) {
    using namespace binomial;

    using sgs_fixed_n_default_t =
        SimGlobalStateFixedNDefault<gen_t, value_t, uint_t, gr_t>;
    using kbs_fixed_n_default_t = KevlarBoundStateFixedNDefault<gr_t>;

    add_fixed_n_default<sgs_fixed_n_default_t, kbs_fixed_n_default_t>(m);

    using simple_selection_t = SimpleSelection<value_t>;
    add_simple_selection<simple_selection_t, gen_t, value_t, uint_t, gr_t>(m);

    using thompson_t = Thompson<value_t>;
    add_thompson<thompson_t, gen_t, value_t, uint_t, gr_t>(m);

    add_berry_inla_to_module<2>(m, "BerryINLA2", "BerryINLA2SimGlobalState",
                                "BerryINLA2SimState");
    add_berry_inla_to_module<3>(m, "BerryINLA3", "BerryINLA3SimGlobalState",
                                "BerryINLA3SimState");
    add_berry_inla_to_module<4>(m, "BerryINLA4", "BerryINLA4SimGlobalState",
                                "BerryINLA4SimState");
}

/*
 * Adds exponential models.
 */
void add_exponential_to_module(py::module_& m) {
    using namespace exponential;
    using sgs_fixed_n_log_hazard_rate_t =
        SimGlobalStateFixedNLogHazardRate<gen_t, value_t, uint_t, gr_t>;
    using kbs_fixed_n_log_hazard_rate_t =
        KevlarBoundStateFixedNLogHazardRate<gr_t>;
    using simple_log_rank_t = exponential::SimpleLogRank<value_t>;

    add_fixed_n_log_hazard_rate<sgs_fixed_n_log_hazard_rate_t,
                                kbs_fixed_n_log_hazard_rate_t>(m);

    add_simple_log_rank<simple_log_rank_t, gen_t, value_t, uint_t, gr_t>(m);
}

/*
 * Function to add all model classes into module m.
 * Populate this function as more models are exported.
 */
void add_to_module(py::module_& m) {
    using mb_t = ModelBase<value_t>;
    using sgs_t = SimGlobalStateBase<value_t, uint_t>;
    using kbs_t = KevlarBoundStateBase<value_t>;

    add_model_base<mb_t>(m);
    add_sim_global_state_base<sgs_t>(m);
    add_kevlar_bound_state_base<kbs_t>(m);

    using fsas_t = FixedSingleArmSize;
    add_fixed_single_arm_size<fsas_t>(m);

    py::module_ binom_m =
        m.def_submodule("binomial", "Binomial model submodule.");
    add_binomial_to_module(binom_m);

    py::module_ exp_m =
        m.def_submodule("exponential", "Exponential model submodule.");
    add_exponential_to_module(exp_m);
}

}  // namespace model
}  // namespace kevlar
