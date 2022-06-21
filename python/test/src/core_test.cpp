#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <export_utils/types.hpp>
#include <model.hpp>
#include <random>

namespace py = pybind11;

using value_t = imprint::py_double_t;
using uint_t = imprint::py_uint_t;

void add_model_to_module(py::module_& m) {
    using namespace imprint::model;
    using sgs_t = SimGlobalStateBase<value_t, uint_t>;
    add_base_tests<sgs_t>(m);
}

PYBIND11_MODULE(core_test, m) {
    py::module_ model_m = m.def_submodule("model", "Model test submodule.");
    add_model_to_module(model_m);
}
