#pragma once
#include <pybind11/pybind11.h>

namespace kevlar {
namespace model {

/* Declarations for adding each model */
// Every such function should be defined in its own cpp
// and called inside add_to_module defined in model.cpp.

void add_binomial_control_k_treatment(pybind11::module_&);

void add_to_module(pybind11::module_&);

} // namespace model
} // namespace kevlar
