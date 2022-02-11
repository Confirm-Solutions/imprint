#include <core.hpp>
#include <model/model.hpp>
#include <util/util.hpp>
#include <std/std.hpp>

PYBIND11_MODULE(core, m) {
    /* Call each adder function from each subdirectory */
    kevlar::model::add_to_module(m);
    kevlar::util::add_to_module(m);
    kevlar::std::add_to_module(m);

    /* Rest of the dependencies */
    kevlar::add_intersum(m);
}
