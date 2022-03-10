#include <stats/stats.hpp>

namespace kevlar {
namespace stats {

namespace py = pybind11;

void add_to_module(py::module_& m)
{
    add_inter_sum(m);
    add_upper_bound(m);
}

} // namespace stats
} // namespace kevlar
