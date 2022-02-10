#include <model/model.hpp>
#include <util/util.hpp>

PYBIND11_MODULE(pykevlar, m) {

    using namespace kevlar;

    /* Call each adder function from each subdirectory */
    model::add_to_module(m);
    util::add_to_module(m);

}
