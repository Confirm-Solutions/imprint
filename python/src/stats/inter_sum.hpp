#pragma once
#include <pybind11/pybind11.h>
#include <export_utils/types.hpp>

namespace kevlar {
namespace stats {

namespace py = pybind11;

template <class MSBType
        , class ISType>
void add_inter_sum(pybind11::module_& m)
{
    using msb_t = MSBType;
    using is_t = ISType;
    py::class_<is_t>(m, "InterSum")
        .def(py::init<>())
        .def(py::init<py_size_t, py_size_t, py_size_t>())
        .def("update", &is_t::template update<msb_t&>)
        .def("pool", &is_t::pool)
        .def("reset", &is_t::reset)
        .def("type_I_sum", 
                py::overload_cast<>(&is_t::type_I_sum),
                py::return_value_policy::reference_internal)
        .def("type_I_sum_const", 
                py::overload_cast<>(&is_t::type_I_sum, py::const_),
                py::return_value_policy::reference_internal)
        .def("grad_sum", 
                py::overload_cast<>(&is_t::grad_sum),
                py::return_value_policy::reference_internal)
        .def("grad_sum_const", 
                py::overload_cast<>(&is_t::grad_sum, py::const_),
                py::return_value_policy::reference_internal)
        .def("n_tiles", &is_t::n_tiles)
        .def("n_params", &is_t::n_params)
        .def(py::pickle(
            [](const is_t& p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(
                        p.type_I_sum(),
                        p.grad_sum(), 
                        p.n_params());
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 3) {
                    throw std::runtime_error("Invalid state!");
                }

                using type_I_sum_t = std::decay_t<decltype(std::declval<is_t>().type_I_sum())>;
                using grad_sum_t = std::decay_t<decltype(std::declval<is_t>().grad_sum())>;
                using n_params_t = std::decay_t<decltype(std::declval<is_t>().n_params())>;

                auto type_I_sum = t[0].cast<type_I_sum_t>();
                auto grad_sum = t[1].cast<grad_sum_t>();
                auto n_params = t[2].cast<n_params_t>();

                /* Create a new C++ instance */
                is_t p(type_I_sum.rows(), 
                       type_I_sum.cols(),
                       n_params);
                p.type_I_sum() = type_I_sum;
                p.grad_sum() = grad_sum;

                return p;
            }))
        ;
}

} // namespace stats
} // namespace kevlar
