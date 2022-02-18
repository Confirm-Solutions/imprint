#pragma once
#include <pybind11/pybind11.h>

namespace kevlar {
namespace grid {

void add_grid_range(pybind11::module_&);
void add_gridder(pybind11::module_&);
void add_adagrid_internal(pybind11::module_&);

void add_to_module(pybind11::module_&);

} // namespace grid
} // namespace kevlar
