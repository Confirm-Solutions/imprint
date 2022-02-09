#include <pybind/model/model.hpp>
#include <kevlar_bits/model/binomial_control_k_treatment.hpp>
#include <kevlar_bits/util/range/grid_range.hpp>

namespace kevlar {
namespace model {

namespace py = pybind11;

void add_binomial_control_k_treatment(py::module_& m)
{
    using bckt_t = kevlar::BinomialControlkTreatment<double, int>;
    py::class_<bckt_t>(m, "BinomialControlkTreatment")
        .def(py::init<
                size_t,
                size_t,
                size_t,
                const kevlar::GridRange<double, int>&,
                double>())
        .def("n_gridpts", &bckt_t::n_gridpts)
        .def("tr_cov", &bckt_t::tr_cov)
        .def("tr_max_cov", &bckt_t::tr_max_cov)
        ;
}

} // namespace model
} // namespace kevlar
