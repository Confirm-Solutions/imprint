#include <util/range/range.hpp>

namespace kevlar {
namespace util {
namespace range {

void add_to_module(pybind11::module_& m)
{
    add_grid_range(m);
}

} // namespace range
} // namespace util
} // namespace kevlar
