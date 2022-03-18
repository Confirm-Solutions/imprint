#include <std/std.hpp>
#include <std/random/random.hpp>

namespace kevlar {
namespace std {

void add_to_module(pybind11::module_& m) { random::add_to_module(m); }

}  // namespace std
}  // namespace kevlar
