#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <stats/stats.hpp>
#include <stats/inter_sum.hpp>
#include <stats/upper_bound.hpp>
#include <kevlar_bits/stats/upper_bound.hpp>
#include <kevlar_bits/stats/inter_sum.hpp>
#include <kevlar_bits/grid/grid_range.hpp>
#include <kevlar_bits/grid/tile.hpp>
#include <kevlar_bits/model/base.hpp>
#include <export_utils/types.hpp>

namespace kevlar {
namespace stats {

namespace py = pybind11;

void add_to_module(py::module_& m) {
    using tile_t = Tile<py_double_t>;
    using gr_t = GridRange<py_double_t, py_uint_t, tile_t>;
    using mb_t = ModelBase<py_double_t, py_uint_t, gr_t>;
    using msb_t = ModelStateBase<py_double_t, py_uint_t, gr_t>;
    using is_t = InterSum<py_double_t, py_uint_t>;
    using ub_t = UpperBound<py_double_t>;

    add_inter_sum<msb_t, is_t>(m);
    add_upper_bound<gr_t, mb_t, is_t, ub_t>(m);
}

}  // namespace stats
}  // namespace kevlar
