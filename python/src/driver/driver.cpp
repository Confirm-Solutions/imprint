#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>
#include <driver/driver.hpp>
#include <driver/fit.hpp>
#include <export_utils/types.hpp>
#include <kevlar_bits/grid/tile.hpp>
#include <kevlar_bits/grid/grid_range.hpp>
#include <kevlar_bits/model/base.hpp>
#include <kevlar_bits/stats/inter_sum.hpp>
#include <random>

namespace kevlar {
namespace driver {

namespace py = pybind11;

void add_to_module(py::module_& m) {
    using gen_t = std::mt19937;
    using tile_t = Tile<py_double_t>;
    using gr_t = GridRange<py_double_t, py_uint_t, tile_t>;
    using model_t = ModelBase<py_double_t, py_uint_t, gr_t>;
    using is_t = InterSum<py_double_t, py_uint_t>;

    add_fit<gen_t, model_t, gr_t, is_t>(m);
}

}  // namespace driver
}  // namespace kevlar
