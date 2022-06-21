#pragma once
#include <cstddef>

namespace imprint {
namespace grid {

template <class ValueType>
struct Tile;

template <class ValueType, class UIntType, class TileType>
struct GridRange;

struct Gridder;

template <class ValueType>
struct HyperPlaneView;

template <class ValueType>
struct HyperPlane;

}  // namespace grid
}  // namespace imprint
