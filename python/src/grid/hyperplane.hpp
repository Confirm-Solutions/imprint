#pragma once
#include <pybind11/pybind11.h>

#include <imprint_bits/util/types.hpp>

namespace imprint {
namespace grid {

namespace py = pybind11;

template <class HyperPlaneType>
void add_hyperplane(py::module_& m) {
    using hp_t = HyperPlaneType;
    using value_t = typename hp_t::value_t;
    py::class_<hp_t>(m, "HyperPlane")
        .def(py::init<const Eigen::Ref<const colvec_type<value_t>>,
                      const value_t&>(),
             py::arg("normal"),
             py::arg("shift"))
        .def(py::pickle(
            [](const hp_t& p) {  // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                colvec_type<value_t> n = p.normal();
                return py::make_tuple(n, p.shift());
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 2) {
                    throw std::runtime_error("Invalid state!");
                }

                using n_t = colvec_type<value_t>;
                using s_t =
                    std::decay_t<decltype(std::declval<hp_t>().shift())>;

                auto normal = t[0].cast<n_t>();
                auto shift = t[1].cast<s_t>();

                /* Create a new C++ instance */
                hp_t p(normal, shift);

                return p;
            }));
}

}  // namespace grid
}  // namespace imprint
