#pragma once
#include <imprint_bits/util/d_ary_int.hpp>
#include <imprint_bits/util/macros.hpp>
#include <imprint_bits/util/types.hpp>

namespace imprint {
namespace grid {

/*
 * Finds the orientation of v w.r.t. sf.
 * Simply delegates to sf.find_orient.
 */
template <class VecType, class SurfType>
IMPRINT_STRONG_INLINE auto find_orient(const VecType& v, const SurfType& sf) {
    return sf.find_orient(v);
}

namespace internal {

/*
 * Returns true if the max tile, i.e. the
 * rectangular tile defined by the center
 * and radius, is on one side of the surface.
 * Note that such a tile may have corners on the surface.
 *
 * @param   tile    Tile object.
 * @param   sf      Surface object.
 * @param   save_orient functor that saves the reason for return value.
 *                      If the max tile is on one side of sf (returns true),
 *                      reason will be set to either
 *                      orient_type::non_neg, orient_type::non_pos,
 * orient_type::on, depending on if it is in the non-negative, non-positive, or
 * boundary orientation. Otherwise, it will be set to orient_type::none.
 *
 * @return  true if max tile is on one side of sf.
 */
template <class TileType, class SurfaceType, class SaveOrientType,
          class IterType>
IMPRINT_STRONG_INLINE bool is_oriented_(const TileType& tile,
                                        const SurfaceType& sf,
                                        SaveOrientType save_orient,
                                        IterType begin, IterType end) {
    size_t n_pos = 0;
    size_t n_neg = 0;

    for (; begin != end; ++begin) {
        const auto& v = *begin;

        // side will be one of: pos, neg, on.
        auto side = find_orient(v, sf);

        n_pos += (side == orient_type::pos);
        n_neg += (side == orient_type::neg);

        // if both are positive, not regular
        if (n_pos && n_neg) {
            save_orient(orient_type::none);
            return false;
        }
    }

    // Note: one of n_pos or n_neg must be 0
    auto ori = (n_pos > 0)
                   ? orient_type::non_neg
                   : ((n_neg > 0) ? orient_type::non_pos : orient_type::on);
    save_orient(ori);
    return true;
}

template <class TileType, class SurfaceType, class SaveOrientType>
IMPRINT_STRONG_INLINE bool is_oriented_(const TileType& tile,
                                        const SurfaceType& sf,
                                        SaveOrientType save_orient) {
    if (tile.is_regular()) {
        return is_oriented_(tile, sf, save_orient, tile.begin_full(),
                            tile.end_full());
    } else {
        return is_oriented_(tile, sf, save_orient, tile.begin(), tile.end());
    }
}

}  // namespace internal

/*
 * Returns true if the tile
 * is on one side of the surface sf.
 */
template <class TileType, class SurfaceType>
IMPRINT_STRONG_INLINE bool is_oriented(const TileType& tile,
                                       const SurfaceType& sf) {
    return internal::is_oriented_(tile, sf, [](auto) {});
}

/*
 * Same as above and additionally records into "reason" the
 * orientation of the tile w.r.t. sf.
 */
template <class TileType, class SurfaceType>
IMPRINT_STRONG_INLINE bool is_oriented(const TileType& tile,
                                       const SurfaceType& sf,
                                       orient_type& reason) {
    return internal::is_oriented_(tile, sf, [&](orient_type r) { reason = r; });
}

/*
 * Computes the intersection of tile and surface sf.
 * After the function call, nn_tile will be updated with
 * the new vertices such that it is non-negatively oriented,
 * and np_tile will be non-positively oriented.
 * This function assumes that surf will intersect tile
 * in their respective geometric sense.
 *
 * TODO: currently for simplicity, if tile is not regular,
 * we simply copy that structure as both p_tile and n_tile.
 */
template <class TileType, class SurfType>
void intersect(const TileType& tile, const SurfType& surf, TileType& p_tile,
               TileType& n_tile) {
    using tile_t = std::decay_t<TileType>;
    using value_t = typename tile_t::value_t;

    // if not regular, copy to both output tiles
    if (!tile.is_regular()) {
        p_tile = tile;
        n_tile = tile;
        return;
    }

    // clear the contents of output tiles
    // before appending vertices.
    p_tile.clear();
    n_tile.clear();

    auto n_params = tile.n_params();
    const auto& radius = tile.radius();

    colvec_type<value_t> v_new(n_params);
    colvec_type<value_t> dir(n_params);
    dir.setZero();

    auto it = tile.begin_full();
    for (; it != tile.end_full(); ++it) {
        const auto& v = *it;
        auto v_ori = find_orient(v, surf);

        // append the corner to the correct output tile(s).
        // this handles all updates for the existing vertices.
        if (v_ori == orient_type::pos) {
            p_tile.emplace_back(v);
        } else if (v_ori == orient_type::on) {
            p_tile.emplace_back(v);
            n_tile.emplace_back(v);
        } else if (v_ori == orient_type::neg) {
            n_tile.emplace_back(v);
        } else {
            throw std::runtime_error("Unexpected orientation type.");
        }

        const auto& bits = it.bits()();  // actual underlying bit array

        // iterate through all neighboring vertices
        // such that current vertex is k-lower, i.e.
        // the kth entry is lower than that of the neighboring vertex,
        // and add any intersected points that are not current vertices.
        for (int k = 0; k < n_params; ++k) {
            // if not k-lower
            if (bits[k]) continue;

            // set current positive direction
            dir[k] = 2 * radius[k];

            // intersection = v + alpha * dir
            value_t alpha = surf.intersect(v, dir);

            // if valid intersection
            if (0 < alpha && alpha < 1) {
                // append to both tiles
                v_new = v + alpha * dir;
                p_tile.emplace_back(v_new);
                n_tile.emplace_back(v_new);
            }

            // unset current direction
            dir[k] = 0;
        }
    }
}

}  // namespace grid
}  // namespace imprint
