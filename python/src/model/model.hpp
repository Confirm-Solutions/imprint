#pragma once
#include <pybind11/pybind11.h>

namespace kevlar {
namespace model {

void add_to_module(pybind11::module_&);

}  // namespace model
}  // namespace kevlar
