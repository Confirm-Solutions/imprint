#pragma once
#include <cstddef>

namespace kevlar {

// TODO: make into user-configurable values.
inline constexpr size_t l1_cache_size = 1<<18;
inline constexpr size_t l2_cache_size = 1<<22;

} // namespace kevlar
