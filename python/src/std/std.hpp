#pragma once
#include <pybind11/pybind11.h>

namespace kevlar {
namespace std {

void add_to_module(pybind11::module_&);

}  // namespace std
}  // namespace kevlar
