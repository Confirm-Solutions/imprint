#include <std/random/random.hpp>
#include <std/random/mt19937.hpp>
#include <random>

namespace kevlar {
namespace std {
namespace random {

void add_to_module(pybind11::module_& m)
{
    add_mt19937<::std::mt19937>(m); 
}

} // namespace random
} // namespace std
} // namespace kevlar
