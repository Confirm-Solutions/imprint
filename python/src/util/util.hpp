#pragma once
#include <pybind11/pybind11.h>

namespace kevlar {
namespace util {

void add_gridder(pybind11::module_&);
void add_to_module(pybind11::module_&);

} // namespace util
} // namespace kevlar
