#include <core.hpp>
#include <driver/driver.hpp>
#include <grid/grid.hpp>
#include <stats/stats.hpp>
#include <model/model.hpp>
#include <std/std.hpp>

PYBIND11_MODULE(core, m) {
    /* Call each adder function from each subdirectory */
    kevlar::model::add_to_module(m);
    kevlar::grid::add_to_module(m);
    kevlar::stats::add_to_module(m);
    kevlar::std::add_to_module(m);
    kevlar::driver::add_to_module(m);

    /* Rest of the dependencies */
}
