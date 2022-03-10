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
    add_model_base(m);
    add_model_state_base(m);
    add_control_k_treatment_base(m);
    add_binomial_control_k_treatment(m);
    add_exp_control_k_treatment(m);
}

} // namespace model
} // namespace kevlar
