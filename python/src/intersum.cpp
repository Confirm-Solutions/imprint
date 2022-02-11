#include <core.hpp>
#include <pybind11/eigen.h>
#include <kevlar_bits/model/base.hpp>
#include <kevlar_bits/intersum.hpp>

namespace kevlar {

namespace py = pybind11;

void add_intersum(pybind11::module_& m)
{
    using msb_t = ModelStateBase<double, uint32_t>;
    using is_t = InterSum<double, uint32_t>;
    py::class_<is_t>(m, "InterSum")
        .def(py::init<>())
        .def(py::init<size_t, size_t, size_t>())
        .def("update", &is_t::update<msb_t&>)
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
        .def("n_accum", 
                py::overload_cast<>(&is_t::n_accum),
                py::return_value_policy::reference_internal)
        .def("n_accum_const", 
                py::overload_cast<>(&is_t::n_accum, py::const_))
        .def(py::pickle(
            [](const is_t& p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(
                        p.type_I_sum(),
                        p.grad_sum(), 
                        p.n_accum(), 
                        p.n_params());
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 4) {
                    throw std::runtime_error("Invalid state!");
                }

                using type_I_sum_t = std::decay_t<decltype(std::declval<is_t>().type_I_sum())>;
                using grad_sum_t = std::decay_t<decltype(std::declval<is_t>().grad_sum())>;
                using n_accum_t = std::decay_t<decltype(std::declval<is_t>().n_accum())>;
                using n_params_t = std::decay_t<decltype(std::declval<is_t>().n_params())>;

                auto type_I_sum = t[0].cast<type_I_sum_t>();
                auto grad_sum = t[1].cast<grad_sum_t>();
                auto n_accum = t[2].cast<n_accum_t>();
                auto n_params = t[3].cast<n_params_t>();

                /* Create a new C++ instance */
                is_t p(type_I_sum.rows(), 
                       type_I_sum.cols(),
                       n_params);
                p.type_I_sum() = type_I_sum;
                p.grad_sum() = grad_sum;
                p.n_accum() = n_accum;

                return p;
            }))
        ;
}

} // namespace kevlar
