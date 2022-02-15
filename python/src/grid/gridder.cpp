#include <grid/grid.hpp>
#include <pybind11/eigen.h>
#include <kevlar_bits/grid/gridder.hpp>

namespace kevlar {
namespace grid {

namespace py = pybind11;

colvec_type<double> make_grid_wrap(size_t n, double l, double u)
{
    return Gridder::make_grid(n, l, u);
}

void add_gridder(pybind11::module_& m)
{
    using gridder_t = Gridder;
    py::class_<gridder_t>(m, "Gridder")
        .def(py::init<>())
        .def("radius", &gridder_t::radius<double>)
        .def("make_grid", &make_grid_wrap)
        ;
}

} // namespace grid
} // namespace kevlar
