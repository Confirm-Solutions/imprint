#pragma once
#include <pybind11/pybind11.h>

namespace kevlar {
namespace std {
namespace random {

void add_mt19937(pybind11::module_&);

void add_to_module(pybind11::module_&);

} // namespace random
} // namespace std
} // namespace kevlar
