#pragma once
#include <imprint_bits/grid/decl.hpp>
#include <imprint_bits/grid/utils.hpp>
#include <imprint_bits/util/macros.hpp>
#include <imprint_bits/util/types.hpp>
#include <type_traits>
#include <vector>

namespace imprint {
namespace grid {

template <class ValueType, class UIntType, class TileType>
struct GridRange {
    using value_t = ValueType;
    using uint_t = UIntType;
    using tile_t = TileType;
    using bits_t = unsigned char;  // TODO: generalize?

   private:
    mat_type<value_t> thetas_;       // matrix of theta vectors
    mat_type<value_t> radii_;        // matrix of radius vectors
    colvec_type<uint_t> sim_sizes_;  // vector of simulation sizes

    // updated via member functions
    std::vector<uint_t> cum_n_tiles_;  // cum_n_tiles_[i] = cumulative number of
                                       // tiles for ith gridpoint
    std::vector<bits_t> bits_;  // vector of bits to represent ISH of each tile
    std::vector<tile_t>
        tiles_;  // vector of tiles (flattened across all gridpoints)

    bits_t all_alt_bits_;

    IMPRINT_STRONG_INLINE
    static constexpr bits_t compute_init_bits(size_t max_bits) { return 0; }

    IMPRINT_STRONG_INLINE
    static constexpr bits_t compute_all_alt_bits(size_t max_bits) {
        bits_t out = 0;
        bits_t pos = 1;
        for (size_t b = 0; b < max_bits; ++b, pos <<= 1) {
            out |= pos;
        }
        return out;
    }

    IMPRINT_STRONG_INLINE
    void set_null(bits_t& bits, size_t hypo, bool is_null = true) {
        unsigned char t = (static_cast<unsigned char>(1) << hypo);
        auto true_bit = -is_null;
        bits = ((~true_bit) & (bits | t)) | (true_bit & (bits & (~t)));
    }

    IMPRINT_STRONG_INLINE
    bool is_all_alt(bits_t bits) const {
        return all_alt_bits_ && (bits == all_alt_bits_);
    }

    void reset_tiles_viewer() {
        // if tiles haven't been created yet
        if (tiles_.size() == 0) return;

        size_t pos = 0;
        for (size_t i = 0; i < n_gridpts(); ++i) {
            for (size_t j = 0; j < n_tiles(i); ++j, ++pos) {
                tiles_[pos].center(thetas_.col(i));
                tiles_[pos].radius(radii_.col(i));
            }
        }
    }

   public:
    GridRange() = default;

    GridRange(uint_t dim, uint_t size)
        : thetas_(dim, size), radii_(dim, size), sim_sizes_(size) {}

    GridRange(const Eigen::Ref<const mat_type<value_t>>& thetas,
              const Eigen::Ref<const mat_type<value_t>>& radii,
              const Eigen::Ref<const colvec_type<uint_t>>& sim_sizes)
        : thetas_(thetas), radii_(radii), sim_sizes_(sim_sizes) {}

    template <class VecSurfType>
    GridRange(const Eigen::Ref<const mat_type<value_t>>& thetas,
              const Eigen::Ref<const mat_type<value_t>>& radii,
              const Eigen::Ref<const colvec_type<uint_t>>& sim_sizes,
              const VecSurfType& surfs, bool do_prune = true)
        : thetas_(thetas), radii_(radii), sim_sizes_(sim_sizes) {
        create_tiles(surfs);
        if (do_prune) prune();
    }

    GridRange(const GridRange& gr)
        : thetas_(gr.thetas_),
          radii_(gr.radii_),
          sim_sizes_(gr.sim_sizes_),
          cum_n_tiles_(gr.cum_n_tiles_),
          bits_(gr.bits_),
          tiles_(gr.tiles_) {
        reset_tiles_viewer();
    }

    GridRange(GridRange&& gr)
        : thetas_(std::move(gr.thetas_)),
          radii_(std::move(gr.radii_)),
          sim_sizes_(std::move(gr.sim_sizes_)),
          cum_n_tiles_(std::move(gr.cum_n_tiles_)),
          bits_(std::move(gr.bits_)),
          tiles_(std::move(gr.tiles_)) {
        reset_tiles_viewer();
    }

    GridRange& operator=(const GridRange& gr) {
        thetas_ = gr.thetas_;
        radii_ = gr.radii_;
        sim_sizes_ = gr.sim_sizes_;
        cum_n_tiles_ = gr.cum_n_tiles_;
        bits_ = gr.bits_;
        tiles_ = gr.tiles_;
        reset_tiles_viewer();
        return *this;
    }

    GridRange& operator=(GridRange&& gr) {
        thetas_ = std::move(gr.thetas_);
        radii_ = std::move(gr.radii_);
        sim_sizes_ = std::move(gr.sim_sizes_);
        cum_n_tiles_ = std::move(gr.cum_n_tiles_);
        bits_ = std::move(gr.bits_);
        tiles_ = std::move(gr.tiles_);
        reset_tiles_viewer();
        return *this;
    }

    /*
     * Creates the tile information based on current values of
     * gridpoints and radii information.
     *
     * It is undefined behavior if gridpoints and radii are not set.
     *
     * @param   vec_surf    vector of surface objects.
     *                      vec_surf[i] corresponds to the surface that
     *                      divides the parameter space to get ith null
     * hypothesis space. Assumed that the non-negative side of the surface is
     *                      the null-hypothesis region.
     */
    template <class VecSurfaceType>
    void create_tiles(const VecSurfaceType& vec_surf) {
        cum_n_tiles_.resize(n_gridpts() + 1);
        cum_n_tiles_[0] = 0;

        bits_.reserve(n_gridpts());
        tiles_.reserve(
            n_gridpts());  // slight optimization
                           // we know we need at least 1 for each gridpoint.

        const size_t max_bits = vec_surf.size();  // max number of bits allowed
        assert(max_bits <= sizeof(bits_t) * 8);

        // this represents all alternative hypothesis being true
        // note that there may be some padded bits which are
        // set to null hypothesis being true,
        // so if max_bits < sizeof(bits_t) * 8, this value is non-trivial.
        all_alt_bits_ = compute_all_alt_bits(max_bits);

        // this represents all null-hypothesis being true.
        const bits_t init_bits = compute_init_bits(max_bits);

        size_t tiles_begin = 0;  // begin position of tiles_ for gridpt j
        for (int j = 0; j < thetas_.cols(); ++j) {
            auto theta_j = thetas_.col(j);
            auto radius_j = radii_.col(j);

            // start the queue of tiles with one (regular) tile
            bits_.emplace_back(init_bits);  // sets current bit to init_bits
            tiles_.emplace_back(theta_j, radius_j);

            for (size_t s = 0; s < vec_surf.size(); ++s) {
                const auto& surf = vec_surf[s];
                size_t q_size = tiles_.size() - tiles_begin;

                // iterate over current queue of tiles for current gridpt
                for (size_t i = 0; i < q_size; ++i) {
                    // if tile is on one side of surface
                    orient_type ori;
                    if (is_oriented(tiles_[tiles_begin + i], surf, ori)) {
                        set_null(bits_[tiles_begin + i], s,
                                 (ori == orient_type::non_neg));
                        continue;
                    }

                    // add new (regular) tile
                    bits_.emplace_back();
                    tiles_.emplace_back(theta_j, radius_j);

                    auto& c_bits = bits_[tiles_begin + i];
                    auto& tile =
                        tiles_[tiles_begin + i];  // get ref here because
                                                  // previous emplace_back may
                                                  // invalidate any prior refs.
                    auto& n_bits = bits_.back();
                    auto& n_tile = tiles_.back();
                    auto p_tile = n_tile;

                    // copy ISH of tile into the new tiles
                    n_bits = c_bits;

                    // split the current tile via surf into two smaller tiles
                    //  - p_tile will be oriented non-negatively (surf null hyp
                    //  space)
                    //  - n_tile will be oriented non-positively
                    intersect(tile, surf, p_tile, n_tile);
                    tile = std::move(p_tile);

                    // update ISH for the new tiles
                    set_null(c_bits, s, true);
                    set_null(n_bits, s, false);
                }
            }

            cum_n_tiles_[j + 1] = tiles_.size();
            tiles_begin += cum_n_tiles_[j + 1] - tiles_begin;
        }

        assert(tiles_begin == n_tiles());
    }

    /*
     * Prunes out gridpts and tiles where the ISH is all 0.
     * These correspond to totally alternative regions
     * where we should not even compute Type I error since no null is ever true.
     */
    void prune() {
        if (n_tiles() == 0) return;

        std::vector<uint_t> grid_idx;
        std::vector<uint_t> new_cum_n_tiles;
        std::vector<bits_t> new_bits;
        std::vector<tile_t> new_tiles;

        new_cum_n_tiles.reserve(n_gridpts() + 1);
        new_cum_n_tiles.push_back(0);
        new_bits.reserve(bits_.size());
        new_tiles.reserve(tiles_.size());

        size_t pos = 0;
        for (size_t g = 0; g < n_gridpts(); ++g) {
            size_t n_append = 0;
            for (size_t j = 0; j < n_tiles(g); ++j) {
                const auto& tile = tiles_[pos + j];
                auto bi = bits_[pos + j];
                if (is_all_alt(bi)) continue;
                ++n_append;
                new_bits.emplace_back(bi);
                new_tiles.emplace_back(std::move(tile));
            }
            if (n_append == 0) {
                grid_idx.push_back(g);
            } else {
                new_cum_n_tiles.push_back(n_append + new_cum_n_tiles.back());
            }
            pos += n_tiles(g);
        }

        std::swap(bits_, new_bits);
        std::swap(cum_n_tiles_, new_cum_n_tiles);
        std::swap(tiles_, new_tiles);

        mat_type<value_t> new_thetas(thetas_.rows(),
                                     thetas_.cols() - grid_idx.size());
        mat_type<value_t> new_radii(radii_.rows(),
                                    radii_.cols() - grid_idx.size());
        colvec_type<uint_t> new_sim_sizes(sim_sizes_.size() - grid_idx.size());
        {
            std::sort(grid_idx.begin(), grid_idx.end());
            int nj = 0;
            for (int j = 0; j < thetas_.cols(); ++j) {
                // if current column should be removed
                if (std::binary_search(grid_idx.begin(), grid_idx.end(), j))
                    continue;
                new_thetas.col(nj) = thetas_.col(j);
                new_radii.col(nj) = radii_.col(j);
                new_sim_sizes(nj) = sim_sizes_(j);
                ++nj;
            }
        }
        thetas_.swap(new_thetas);
        radii_.swap(new_radii);
        sim_sizes_.swap(new_sim_sizes);

        // make sure to reset the viewers for the tile objects!
        reset_tiles_viewer();
    }

    /*
     * If these internal members' shapes are changed,
     * user MUST call create_tiles() before using any tile information again.
     */
    IMPRINT_STRONG_INLINE mat_type<value_t>& thetas() { return thetas_; }
    IMPRINT_STRONG_INLINE const mat_type<value_t>& thetas() const {
        return thetas_;
    }
    IMPRINT_STRONG_INLINE mat_type<value_t>& radii() { return radii_; }
    IMPRINT_STRONG_INLINE const mat_type<value_t>& radii() const {
        return radii_;
    }
    IMPRINT_STRONG_INLINE colvec_type<uint_t>& sim_sizes() {
        return sim_sizes_;
    }
    IMPRINT_STRONG_INLINE const colvec_type<uint_t>& sim_sizes() const {
        return sim_sizes_;
    }

    IMPRINT_STRONG_INLINE const std::vector<uint_t>& cum_n_tiles() const {
        return cum_n_tiles_;
    }

    // This function is only valid once create_tiles() has been called.
    IMPRINT_STRONG_INLINE uint_t n_tiles(size_t gridpt_idx) const {
        return cum_n_tiles_[gridpt_idx + 1] - cum_n_tiles_[gridpt_idx];
    }
    IMPRINT_STRONG_INLINE uint_t n_tiles() const { return tiles_.size(); }
    IMPRINT_STRONG_INLINE uint_t n_gridpts() const { return thetas_.cols(); }
    IMPRINT_STRONG_INLINE uint_t n_params() const { return thetas_.rows(); }

    /*
     * Returns true if the tile specified by tile_idx
     * has ISH configuration such that null hypothesis for hypo_idx is true.
     * This function is only valid once create_tiles() has been called.
     * It is well-defined for hypo_idx in the range [0, max_bits()).
     * If create_tiles() were called with a vector of surfaces of size k,
     * then, hypo_idx in the range [k, max_bits()) will return true,
     * i.e. by default, an "empty" hypothesis is assumed to be null.
     */
    IMPRINT_STRONG_INLINE
    bool check_null(size_t tile_idx, size_t hypo_idx) const {
        return (bits_[tile_idx] &
                (static_cast<unsigned char>(1) << hypo_idx)) == 0;
    }

    IMPRINT_STRONG_INLINE
    bool check_null(size_t gridpt_idx, size_t rel_tile_idx,
                    size_t hypo_idx) const {
        size_t tile_idx = n_tiles(gridpt_idx) + rel_tile_idx;
        return check_null(tile_idx, hypo_idx);
    }

    /*
     * Returns true if the gridpoint at idx
     * is associated with a regular tile, i.e. a rectangular tile.
     * This function is only valid once create_tiles() has been called.
     *
     * The note below marked with "XXXX" is about an optimization that has been
     * reverted due to incorrect behavior.
     * XXXX Note: this function originally did:
     * XXXX      return tiles_[tile_idx].is_regular();
     * XXXX but benchmarking shows that there is a MASSIVE speed difference
     * XXXX from the current implementation. Cache is really important...
     * XXXX Idea is that tiles_ is a heterogenous structure which used to
     * contain
     * XXXX std::bitset<> and some Eigen objects.
     * XXXX Iterating through these makes pre-fetching hard
     * XXXX and there are more tiles than gridpoints, so not only does the
     * current
     * XXXX implementation pre-fetch more values at a time,
     * XXXX but also pre-fetches less in total.
     */
    bool is_regular(size_t idx) const {
        return tiles_[cum_n_tiles_[idx]].is_regular();
    }

    IMPRINT_STRONG_INLINE
    static constexpr size_t max_bits() { return sizeof(bits_t) * 8; }

    /*
     * Returns the vector of tiles.
     */
    IMPRINT_STRONG_INLINE const auto& tiles() const { return tiles_; }

    // Helper functions for pickling stuff
    IMPRINT_STRONG_INLINE auto& tiles__() { return tiles_; }
    IMPRINT_STRONG_INLINE auto& cum_n_tiles__() { return cum_n_tiles_; }
    IMPRINT_STRONG_INLINE auto& bits__() { return bits_; }
    IMPRINT_STRONG_INLINE const auto& cum_n_tiles__() const {
        return cum_n_tiles_;
    }
    IMPRINT_STRONG_INLINE const auto& bits__() const { return bits_; }
};

}  // namespace grid
}  // namespace imprint
