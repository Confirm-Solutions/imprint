#include <pybind11/eigen.h>

#include <bound/bound.hpp>
#include <core.hpp>
#include <driver/driver.hpp>
#include <grid/grid.hpp>
#include <imprint_bits/util/types.hpp>
#include <model/model.hpp>
#include <random>

namespace py = pybind11;

PYBIND11_MODULE(core, m) {
    using namespace imprint;

    /* Call each adder function from each subdirectory */
    py::module_ model_m = m.def_submodule("model", "Model submodule.");
    model::add_to_module(model_m);

    py::module_ grid_m = m.def_submodule("grid", "Grid submodule.");
    grid::add_to_module(grid_m);

    py::module_ driver_m = m.def_submodule("driver", "Driver submodule.");
    driver::add_to_module(driver_m);

    py::module_ bound_m = m.def_submodule("bound", "Bound submodule.");
    bound::add_to_module(bound_m);
    /* Rest of the dependencies */

    py::class_<std::mt19937>(m, "mt19937")
        .def(py::init<uint32_t>())
        .def("uniform_sample",
             [](std::mt19937& gen, Eigen::Ref<colvec_type<double>>& out_arr) {
                 std::uniform_real_distribution<double> unif_;
                 size_t n_samples = out_arr.size();
                 for (size_t i = 0; i < n_samples; i++) {
                     out_arr[i] = unif_(gen);
                 }
             });
}
