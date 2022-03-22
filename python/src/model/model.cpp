#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <model/model.hpp>
#include <model/base.hpp>
#include <model/binomial_control_k_treatment.hpp>
#include <model/exp_control_k_treatment.hpp>
#include <export_utils/types.hpp>
#include <kevlar_bits/model/base.hpp>
#include <kevlar_bits/model/binomial_control_k_treatment.hpp>
#include <kevlar_bits/model/exp_control_k_treatment.hpp>
#include <kevlar_bits/grid/tile.hpp>
#include <kevlar_bits/grid/grid_range.hpp>

namespace kevlar {
namespace model {

namespace py = pybind11;

/*
 * Function to add all model classes into module m.
 * Populate this function as more models are exported.
 */
void add_to_module(py::module_& m) {
    using tile_t = Tile<py_double_t>;
    using gr_t = GridRange<py_double_t, py_uint_t, tile_t>;
    using mb_t = ModelBase<py_double_t, py_uint_t, gr_t>;
    using msb_t = ModelStateBase<py_double_t, py_uint_t, gr_t>;
    using cktb_t = ControlkTreatmentBase;
    using bckt_t = BinomialControlkTreatment<py_double_t, py_uint_t, gr_t>;
    using eckt_t = ExpControlkTreatment<py_double_t, py_uint_t, gr_t>;

    add_model_base<mb_t>(m);
    add_model_state_base<msb_t>(m);
    add_control_k_treatment_base<cktb_t>(m);
    add_binomial_control_k_treatment<bckt_t>(m);
    add_exp_control_k_treatment<eckt_t>(m);
}

}  // namespace model
}  // namespace kevlar
