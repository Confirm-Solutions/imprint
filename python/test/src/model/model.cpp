#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <export_utils/types.hpp>
#include <kevlar_bits/model/base.hpp>
#include <model/base.hpp>
#include <random>

namespace kevlar {
namespace model {

namespace py = pybind11;

using value_t = py_double_t;
using uint_t = py_uint_t;
using gen_t = std::mt19937;

void add_to_module(py::module_& m) {
    using sgs_t = SimGlobalStateBase<gen_t, value_t, uint_t>;
    add_base_tests<sgs_t>(m);
}

}  // namespace model
}  // namespace kevlar
