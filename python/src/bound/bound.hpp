#pragma once
#include <pybind11/pybind11.h>

namespace kevlar {
namespace bound {

void add_to_module(pybind11::module_&);

}  // namespace bound
}  // namespace kevlar
