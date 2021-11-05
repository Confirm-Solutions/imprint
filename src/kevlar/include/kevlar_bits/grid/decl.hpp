#pragma once
#include <cstddef>

namespace kevlar {

template <class ValueType, size_t NBits = 8>
struct Tile;

template <class ValueType, class UIntType, class TileType>
struct GridRange;

struct Gridder;

template <class ValueType>
struct HyperPlaneView;

template <class ValueType>
struct HyperPlane;

}  // namespace kevlar
