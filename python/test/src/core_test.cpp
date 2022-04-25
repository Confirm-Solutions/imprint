#include <model/model.hpp>

namespace py = pybind11;

PYBIND11_MODULE(core_test, m) {
    py::module_ model_m = m.def_submodule("model", "Model test submodule.");
    kevlar::model::add_to_module(model_m);
}
