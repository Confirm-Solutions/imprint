#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <export_utils/types.hpp>
#include <imprint_bits/grid/grid_range.hpp>
#include <imprint_bits/grid/tile.hpp>
#include <imprint_bits/model/base.hpp>
#include <imprint_bits/model/binomial/simple_selection.hpp>
#include <imprint_bits/model/binomial/thompson.hpp>
#include <imprint_bits/model/exponential/simple_log_rank.hpp>
#include <model/base.hpp>
#include <model/binomial/fixed_n_default.hpp>
#include <model/binomial/simple_selection.hpp>
#include <model/binomial/thompson.hpp>
#include <model/exponential/fixed_n_log_hazard_rate.hpp>
#include <model/exponential/simple_log_rank.hpp>
#include <model/fixed_single_arm_size.hpp>
#include <model/model.hpp>

namespace imprint {
namespace model {

namespace py = pybind11;

using value_t = py_double_t;
using uint_t = py_uint_t;
using gen_t = std::mt19937;
using tile_t = grid::Tile<value_t>;
using gr_t = grid::GridRange<value_t, uint_t, tile_t>;

/*
 * Adds binomial models.
 */
void add_binomial_to_module(py::module_& m) {
    using namespace binomial;

    using sgs_fixed_n_default_t =
        SimGlobalStateFixedNDefault<gen_t, value_t, uint_t, gr_t>;
    using kbs_fixed_n_default_t = ImprintBoundStateFixedNDefault<gr_t>;

    add_fixed_n_default<sgs_fixed_n_default_t, kbs_fixed_n_default_t>(m);

    using simple_selection_t = SimpleSelection<value_t>;
    add_simple_selection<simple_selection_t, gen_t, value_t, uint_t, gr_t>(m);

    using thompson_t = Thompson<value_t>;
    add_thompson<thompson_t, gen_t, value_t, uint_t, gr_t>(m);
}

/*
 * Adds exponential models.
 */
void add_exponential_to_module(py::module_& m) {
    using namespace exponential;
    using sgs_fixed_n_log_hazard_rate_t =
        SimGlobalStateFixedNLogHazardRate<gen_t, value_t, uint_t, gr_t>;
    using kbs_fixed_n_log_hazard_rate_t =
        ImprintBoundStateFixedNLogHazardRate<gr_t>;
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
    using kbs_t = ImprintBoundStateBase<value_t>;

    add_model_base<mb_t>(m);
    add_sim_global_state_base<sgs_t>(m);
    add_imprint_bound_state_base<kbs_t>(m);

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
}  // namespace imprint
