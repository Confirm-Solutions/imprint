#pragma once
#include <pybind11/pybind11.h>

namespace kevlar {
namespace stats {

void add_to_module(pybind11::module_&);

}  // namespace stats
}  // namespace kevlar
