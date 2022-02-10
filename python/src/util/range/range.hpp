#pragma once
#include <pybind11/pybind11.h>

namespace kevlar {
namespace util {
namespace range {

void add_grid_range(pybind11::module_&);

void add_to_module(pybind11::module_&);

} // namespace range
} // namespace util
} // namespace kevlar
