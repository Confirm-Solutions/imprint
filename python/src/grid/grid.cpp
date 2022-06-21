#include <pybind11/eigen.h>  // must enable for automatic conversion of Eigen
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <export_utils/types.hpp>
#include <grid/adagrid_internal.hpp>
#include <grid/grid.hpp>
#include <grid/grid_range.hpp>
#include <grid/gridder.hpp>
#include <grid/hyperplane.hpp>
#include <grid/tile.hpp>
#include <imprint_bits/bound/typeI_error_bound.hpp>
#include <imprint_bits/grid/adagrid_internal.hpp>
#include <imprint_bits/grid/grid_range.hpp>
#include <imprint_bits/grid/gridder.hpp>
#include <imprint_bits/grid/hyperplane.hpp>
#include <imprint_bits/grid/tile.hpp>

namespace imprint {
namespace grid {

void add_to_module(pybind11::module_& m) {
    using tile_t = Tile<py_double_t>;
    using gr_t = GridRange<py_double_t, py_uint_t, tile_t>;
    using gridder_t = Gridder;
    using adagrid_t = AdaGridInternal;
    using ub_t = bound::TypeIErrorBound<py_double_t>;
    using hp_t = HyperPlane<py_double_t>;
    using vec_surf_t = std::vector<hp_t>;
    using tile_t = Tile<py_double_t>;

    add_gridder<gridder_t, py_double_t, py_uint_t>(m);
    add_tile<tile_t>(m);
    add_grid_range<gr_t, vec_surf_t>(m);
    add_hyperplane<hp_t>(m);
    add_adagrid_internal<adagrid_t, ub_t, gr_t, py_double_t>(m);
}

}  // namespace grid
}  // namespace imprint
