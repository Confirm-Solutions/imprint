#include <grid/grid.hpp>
#include <pybind11/eigen.h> // must enable for automatic conversion of Eigen
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <kevlar_bits/grid/adagrid_internal.hpp>
#include <kevlar_bits/grid/grid_range.hpp>
#include <kevlar_bits/stats/upper_bound.hpp>

namespace kevlar {
namespace grid {

namespace py = pybind11;

void add_adagrid_internal(py::module_& m) 
{
    using ada_t = AdaGridInternal<double>;
    using ub_t = UpperBound<double>;
    using gr_t = GridRange<double, uint32_t>;
    using fn_t = std::function<
        bool(const Eigen::Ref<const colvec_type<double> >&) >;
    py::class_<ada_t>(m, "AdaGridInternal")
        .def(py::init<>())
        .def("update", 
                &ada_t::update<
                    ub_t,
                    gr_t&,
                    gr_t&,
                    fn_t>)
        ;
}

} // namespace grid
} // namespace kevlar
