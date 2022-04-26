#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <bound/bound.hpp>
#include <bound/typeI_error_accum.hpp>
#include <bound/typeI_error_bound.hpp>
#include <export_utils/types.hpp>
#include <kevlar_bits/bound/accumulator/typeI_error_accum.hpp>
#include <kevlar_bits/bound/typeI_error_bound.hpp>
#include <kevlar_bits/grid/grid_range.hpp>
#include <kevlar_bits/grid/tile.hpp>
#include <kevlar_bits/model/base.hpp>

namespace kevlar {
namespace bound {

namespace py = pybind11;

void add_to_module(py::module_& m) {
    using tile_t = grid::Tile<py_double_t>;
    using gr_t = grid::GridRange<py_double_t, py_uint_t, tile_t>;
    using sgs_t = model::SimGlobalStateBase<py_double_t, py_uint_t>;
    using ss_t = typename sgs_t::sim_state_t;
    using kbs_t = model::KevlarBoundStateBase<py_double_t>;
    using acc_t = TypeIErrorAccum<py_double_t, py_uint_t>;
    using kb_t = TypeIErrorBound<py_double_t>;

    add_typeI_error_accum<ss_t, acc_t, gr_t>(m);
    add_typeI_error_bound<gr_t, kbs_t, acc_t, kb_t>(m);
}

}  // namespace bound
}  // namespace kevlar
