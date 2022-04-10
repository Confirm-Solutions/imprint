#include <bound/bound.hpp>
#include <core.hpp>
#include <driver/driver.hpp>
#include <grid/grid.hpp>
#include <model/model.hpp>

namespace py = pybind11;

PYBIND11_MODULE(core, m) {
    /* Call each adder function from each subdirectory */
    py::module_ model_m = m.def_submodule("model", "Model submodule.");
    kevlar::model::add_to_module(model_m);

    py::module_ grid_m = m.def_submodule("grid", "Grid submodule.");
    kevlar::grid::add_to_module(grid_m);
    // kevlar::stats::add_to_module(m);
    // kevlar::driver::add_to_module(m);

    /* Rest of the dependencies */
}
