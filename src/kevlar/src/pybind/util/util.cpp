#include <pybind/util/util.hpp>
#include <pybind/util/range/range.hpp>

namespace kevlar {
namespace util {

void add_to_module(pybind11::module_& m)
{
    range::add_to_module(m);
}

} // namespace util
} // namespace kevlar
