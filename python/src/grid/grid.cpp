#include <grid/grid.hpp>

namespace kevlar {
namespace grid {

void add_to_module(pybind11::module_& m)
{
    add_grid_range(m);
    add_gridder(m);
    add_adagrid_internal(m);
}

} // namespace grid
} // namespace kevlar
