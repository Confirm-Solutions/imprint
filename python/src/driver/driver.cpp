#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <driver/accumulate.hpp>
#include <driver/driver.hpp>
#include <export_utils/types.hpp>
#include <kevlar_bits/bound/accumulator/typeI_error_accum.hpp>
#include <kevlar_bits/grid/grid_range.hpp>
#include <kevlar_bits/grid/tile.hpp>
#include <kevlar_bits/model/base.hpp>

namespace kevlar {
namespace driver {

namespace py = pybind11;

void add_to_module(py::module_& m) {
    using tile_t = grid::Tile<py_double_t>;
    using gr_t = grid::GridRange<py_double_t, py_uint_t, tile_t>;
    using sgs_t = model::SimGlobalStateBase<py_double_t, py_uint_t>;
    using acc_t = bound::TypeIErrorAccum<py_double_t, py_uint_t>;

    add_accumulate<sgs_t, gr_t, acc_t>(m);
}

}  // namespace driver
}  // namespace kevlar
