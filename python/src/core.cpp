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

    py::module_ driver_m = m.def_submodule("driver", "Driver submodule.");
    kevlar::driver::add_to_module(driver_m);

    py::module_ bound_m = m.def_submodule("bound", "Bound submodule.");
    kevlar::bound::add_to_module(bound_m);
    /* Rest of the dependencies */
}
