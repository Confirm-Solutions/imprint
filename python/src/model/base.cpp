#include <model/model.hpp>
#include <pybind11/eigen.h>
#include <kevlar_bits/model/base.hpp>

namespace kevlar {
namespace model {

namespace py = pybind11;

void add_model_base(py::module_& m)
{
    using mb_t = ModelBase<double>;
    py::class_<mb_t>(m, "ModelBase")
        .def("cov_quad", &mb_t::cov_quad)
        .def("max_cov_quad", &mb_t::max_cov_quad)
        ;
}

void add_model_state_base(pybind11::module_& m)
{
    using msb_t = ModelStateBase<double, uint32_t>;
    py::class_<msb_t>(m, "ModelStateBase")
        .def("get_rej_len", &msb_t::get_rej_len)
        .def("get_grad", &msb_t::get_grad)
        ;
}

void add_control_k_treatment_base(pybind11::module_& m)
{
    using ckt_t = ControlkTreatmentBase;
    py::class_<ckt_t>(m, "ControlkTreatmentBase")
        .def(py::init<size_t, size_t, size_t>())
        .def("n_arms", &ckt_t::n_arms)
        .def("ph2_size", &ckt_t::ph2_size)
        .def("n_samples", &ckt_t::n_samples)
        ;
}

} // namespace model
} // namespace kevlar
