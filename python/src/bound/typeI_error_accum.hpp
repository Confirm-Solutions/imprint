#pragma once
#include <pybind11/pybind11.h>

#include <export_utils/types.hpp>
#include <imprint_bits/util/types.hpp>

namespace imprint {
namespace bound {

namespace py = pybind11;

template <class SSType, class AccType, class GridRangeType>
void add_typeI_error_accum(pybind11::module_& m) {
    using ss_t = SSType;
    using acc_t = AccType;
    using grid_range_t = GridRangeType;
    using uint_t = typename acc_t::uint_t;
    py::class_<acc_t>(m, "TypeIErrorAccum")
        .def(py::init<>())
        .def(py::init<py_size_t, py_size_t, py_size_t>(), py::arg("n_models"),
             py::arg("n_tiles"), py::arg("n_params"))
        .def("update",
             &acc_t::template update<Eigen::Ref<const colvec_type<uint_t>>,
                                     ss_t, grid_range_t>,
             py::arg("rej_len"), py::arg("sim_state"), py::arg("grid_range"))
        .def("pool", &acc_t::pool, py::arg("other"))
        .def("pool_raw", &acc_t::pool_raw, py::arg("typeI_sum"),
             py::arg("typeI_score"))
        .def("reset", &acc_t::reset, py::arg("n_models"), py::arg("n_tiles"),
             py::arg("n_params"))
        .def("typeI_sum", py::overload_cast<>(&acc_t::typeI_sum, py::const_),
             py::return_value_policy::reference_internal)
        .def("score_sum", py::overload_cast<>(&acc_t::score_sum, py::const_),
             py::return_value_policy::reference_internal)
        .def("n_tiles", &acc_t::n_tiles)
        .def("n_params", &acc_t::n_params)
        .def("n_models", &acc_t::n_models)
        .def(py::pickle(
            [](const acc_t& p) {  // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(p.typeI_sum(), p.score_sum(),
                                      p.n_params());
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 3) {
                    throw std::runtime_error("Invalid state!");
                }

                using typeI_sum_t =
                    std::decay_t<decltype(std::declval<acc_t>().typeI_sum())>;
                using score_sum_t =
                    std::decay_t<decltype(std::declval<acc_t>().score_sum())>;
                using n_params_t =
                    std::decay_t<decltype(std::declval<acc_t>().n_params())>;

                auto typeI_sum = t[0].cast<typeI_sum_t>();
                auto score_sum = t[1].cast<score_sum_t>();
                auto n_params = t[2].cast<n_params_t>();

                /* Create a new C++ instance */
                acc_t p(typeI_sum.rows(), typeI_sum.cols(), n_params);
                p.typeI_sum__() = typeI_sum;
                p.score_sum__() = score_sum;

                return p;
            }));
}

}  // namespace bound
}  // namespace imprint
