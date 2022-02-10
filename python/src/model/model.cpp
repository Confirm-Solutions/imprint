#include <model/model.hpp>

namespace kevlar {
namespace model {

namespace py = pybind11;

/*
 * Function to add all model classes into module m.
 * Populate this function as more models are exported.
 */
void add_to_module(py::module_& m)
{
    add_binomial_control_k_treatment(m);
}

} // namespace model
} // namespace kevlar
