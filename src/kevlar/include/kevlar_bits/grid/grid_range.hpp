#pragma once
#include <type_traits>
#include <kevlar_bits/util/types.hpp>
#include <kevlar_bits/grid/utils.hpp>

namespace kevlar {

template <class ValueType
        , class UIntType
        , class TileType>
struct GridRange;

template <class ValueType
        , class UIntType>
struct GridptViewer
{
    using value_t = ValueType;
    using uint_t = UIntType;

    GridptViewer(
        size_t dim,
        value_t* ptheta,
        value_t* pradius,
        uint_t* psim_size)
        : theta_(ptheta, dim)
        , radius_(pradius, dim)
        , sim_size_(psim_size)
    {}

    auto& get_theta() { return theta_; }
    auto& get_radius() { return radius_; }
    auto& get_sim_size() { return *sim_size_; }

    void reset(
            value_t* ptheta, 
            value_t* pradius,
            uint_t* psim_size)
    {
        new (&theta_) Eigen::Map<vec_t>(
                ptheta, theta_.size());
        new (&radius_) Eigen::Map<vec_t>(
                pradius, radius_.size());
        sim_size_ = psim_size;
    }

private: 
    using vec_t = std::conditional_t<
            std::is_const_v<value_t>,
            const colvec_type<std::decay_t<value_t> >,
            colvec_type<value_t>
        >;
    Eigen::Map<vec_t> theta_;
    Eigen::Map<vec_t> radius_;
    uint_t* sim_size_;
};

template <class ValueType
        , class UIntType
        , class TileType>
struct GridRange
{
    using value_t = ValueType;
    using uint_t = UIntType;
    using tile_t = TileType;

    struct iterator_type 
    {
        using difference_type = std::ptrdiff_t;
        using value_type = GridptViewer<value_t, uint_t>;
        using pointer = GridptViewer<value_t, uint_t>*;
        using reference = GridptViewer<value_t, uint_t>&;
        using iterator_category = std::random_access_iterator_tag;

        iterator_type(GridRange& outer,
                      size_t cnt)
            : outer_ref_{outer}
            , viewer_(outer.dim(),
                      outer.thetas_.data() + cnt*outer.dim(),
                      outer.radii_.data() + cnt*outer.dim(),
                      outer.sim_sizes_.data()+cnt)
            , cnt_{cnt}
        {}

        iterator_type& operator+=(difference_type n) {
            cnt_ += n;
            auto& outer = outer_ref_.get();
            viewer_.reset(
                    viewer_.get_theta().data() + n*outer.dim(),
                    viewer_.get_radius().data() + n*outer.dim(),
                    outer.sim_sizes_.data() + cnt_);
            return *this;
        }
        iterator_type& operator++() { 
            ++cnt_; 
            auto& outer = outer_ref_.get();
            viewer_.reset(
                    viewer_.get_theta().data() + outer.dim(),
                    viewer_.get_radius().data() + outer.dim(),
                    outer.sim_sizes_.data() + cnt_);
            return *this; 
        }
        reference operator*() { return viewer_; }
        pointer operator->() { return &viewer_; }

        difference_type operator-(const iterator_type& it2)
        {
            return cnt_ - it2.cnt_;
        }

        inline constexpr bool 
        operator==(const iterator_type& it2) const
        { 
            return (this->cnt_ == it2.cnt_) &&
                (&this->outer_ref_.get() == &it2.outer_ref_.get()); 
        }

        inline constexpr bool 
        operator!=(const iterator_type& it2) const
        { 
            return (this->cnt_ != it2.cnt_) ||
                (&this->outer_ref_.get() != &it2.outer_ref_.get()); 
        }

    private:
        std::reference_wrapper<GridRange> outer_ref_;
        GridptViewer<value_t, uint_t> viewer_;
        size_t cnt_;
    };

    struct const_iterator_type 
    {
        using difference_type = std::ptrdiff_t;
        using value_type = GridptViewer<const value_t, const uint_t>;
        using pointer = const GridptViewer<const value_t, const uint_t>*;
        using reference = const GridptViewer<const value_t, const uint_t>&;
        using iterator_category = std::random_access_iterator_tag;

        const_iterator_type(
                const GridRange& outer,
                size_t cnt)
            : outer_ref_{outer}
            , viewer_(outer.dim(),
                      outer.thetas_.data() + cnt*outer.dim(),
                      outer.radii_.data() + cnt*outer.dim(),
                      outer.sim_sizes_.data()+cnt)
            , cnt_{cnt}
        {}

        const_iterator_type& operator+=(difference_type n) {
            cnt_ += n;
            auto& outer = outer_ref_.get();
            viewer_.reset(
                    viewer_.get_theta().data() + n*outer.dim(),
                    viewer_.get_radius().data() + n*outer.dim(),
                    outer.sim_sizes_.data() + cnt_);
            return *this;
        }
        const_iterator_type& operator++() { 
            ++cnt_; 
            const auto& outer = outer_ref_.get();
            viewer_.reset(
                    viewer_.get_theta().data() + outer.dim(),
                    viewer_.get_radius().data() + outer.dim(),
                    outer.sim_sizes_.data() + cnt_);
            return *this; 
        }
        reference operator*() { return viewer_; }
        pointer operator->() { return &viewer_; }

        inline constexpr bool 
        operator==(const const_iterator_type& it2) const
        { 
            return (this->cnt_ == it2.cnt_) &&
                (&this->outer_ref_.get() == &it2.outer_ref_.get()); 
        }

        inline constexpr bool 
        operator!=(const const_iterator_type& it2) const
        { 
            return (this->cnt_ != it2.cnt_) ||
                (&this->outer_ref_.get() != &it2.outer_ref_.get()); 
        }

    private:
        std::reference_wrapper<const GridRange> outer_ref_;
        GridptViewer<const value_t, const uint_t> viewer_;
        size_t cnt_;
    };

    GridRange() =default;

    GridRange(
        uint_t dim,
        uint_t size)
        : thetas_(dim, size)
        , radii_(dim, size)
        , sim_sizes_(size)
    {}

    /*
     * Creates the tile information based on current values of
     * gridpoints and radii information.
     *
     * It is undefined behavior if gridpoints and radii are not set.
     *
     * @param   vec_surf    vector of surface objects.
     *                      vec_surf[i] corresponds to the surface that 
     *                      divides the parameter space to get ith null hypothesis space. 
     *                      Assumed that the non-negative side of the surface is 
     *                      the null-hypothesis region.
     */
    template <class VecSurfaceType>
    void create_tiles(const VecSurfaceType& vec_surf) 
    {
        n_tiles_.resize(n_gridpts());
        tiles_.reserve(n_gridpts());    // slight optimization
                                        // we know we need at least 1 for each gridpoint.

        size_t tiles_begin = 0; // begin position of tiles_ for gridpt j
        for (int j = 0; j < thetas_.cols(); ++j) {
            auto theta_j = thetas_.col(j);
            auto radius_j = radii_.col(j);

            // start the queue of tiles with one (regular) tile
            tiles_.emplace_back(theta_j, radius_j);

            for (size_t s = 0; s < vec_surf.size(); ++s) {
                const auto& surf = vec_surf[s];
                size_t q_size = tiles_.size() - tiles_begin;

                // iterate over current queue of tiles for current gridpt
                for (size_t i = 0; i < q_size; ++i) {
                    // if tile is on one side of surface
                    orient_type ori;
                    if (is_oriented(tiles_[tiles_begin+i], surf, ori)) {
                        tiles_[tiles_begin+i].set_null(
                                s, (ori == orient_type::non_neg));
                        continue;
                    }

                    // add new (regular) tile
                    tiles_.emplace_back(theta_j, radius_j);

                    auto& tile = tiles_[tiles_begin+i]; // get ref here because
                                                        // previous emplace_back may 
                                                        // invalidate any prior refs.
                    auto& n_tile = tiles_.back();
                    auto p_tile = n_tile;

                    // copy ISH of tile into the new tiles
                    n_tile.set_null(tile);
                    p_tile.set_null(tile);

                    // split the current tile via surf into two smaller tiles
                    //  - p_tile will be oriented non-negatively (surf null hyp space)
                    //  - n_tile will be oriented non-positively
                    intersect(tile, surf, p_tile, n_tile);
                    tile = std::move(p_tile);
                    
                    // update ISH for the new tiles
                    tile.set_null(s, true);
                    n_tile.set_null(s, false);
                }
            }

            n_tiles_[j] = tiles_.size() - tiles_begin;
            tiles_begin += n_tiles_[j];
        }
    }

    /*
     * Prunes out gridpts and tiles where the ISH is all 0.
     * These correspond to totally alternative regions
     * where we should not even compute Type I error since no null is ever true.
     */
    void prune()
    {
        std::vector<uint_t> tile_idx;
        std::vector<uint_t> grid_idx;

        size_t pos = 0;
        for (size_t g = 0; g < n_gridpts(); ++g) 
        {
            size_t n_remove = 0;
            for (size_t j = 0; j < n_tiles(g); ++j)
            {
                const auto& tile = tiles_[pos+j];
                bool is_alt = true;
                for (size_t i = 0; i < tile_t::n_bits; ++i) {
                    if (tile.check_null(i)) {
                        is_alt = false;
                        break;
                    }
                }
                if (is_alt) {
                    tile_idx.push_back(pos+j);
                    ++n_remove;
                }
            }
            if (n_remove == n_tiles(g)) {
                grid_idx.push_back(g);
            }
            pos += n_tiles(g);
        }

        mat_type<value_t> new_thetas(
                thetas_.rows(), thetas_.cols()-grid_idx.size());
        mat_type<value_t> new_radii(
                radii_.rows(), radii_.cols()-grid_idx.size());
        colvec_type<uint_t> new_sim_sizes(
                sim_sizes_.size()-grid_idx.size());
        colvec_type<uint_t> new_n_tiles(
                n_tiles_.size()-grid_idx.size());
        {
            int nj = 0;
            for (int j = 0; j < thetas_.cols(); ++j) {
                // if current column should be removed
                if (std::find(grid_idx.begin(), grid_idx.end(), j) 
                        != grid_idx.end()) continue;
                new_thetas.col(nj) = thetas_.col(j);
                new_radii.col(nj) = radii_.col(j);
                new_sim_sizes(nj) = sim_sizes_(j);
                new_n_tiles(nj) = n_tiles_(j);
                ++nj;
            }
        }
        thetas_.swap(new_thetas);
        radii_.swap(new_radii);
        sim_sizes_.swap(new_sim_sizes);
        n_tiles_.swap(new_n_tiles);

        std::vector<tile_t> new_tiles;
        {
            size_t nj = 0;
            for (size_t j = 0; j < tiles_.size(); ++j) {
                if (std::find(tile_idx.begin(), tile_idx.end(), j)
                        != tile_idx.end()) continue;         
                new_tiles.push_back(tiles_[j]);
                ++nj;
            }
        }
        std::swap(tiles_, new_tiles);
    }

    /*
     * If these internal members' shapes are changed,
     * user MUST call create_tiles() before using any tile information again.
     */
    mat_type<value_t>& thetas() { return thetas_; }
    const mat_type<value_t>& thetas() const { return thetas_; }
    mat_type<value_t>& radii() { return radii_; }
    const mat_type<value_t>& radii() const { return radii_; }
    colvec_type<uint_t>& sim_sizes() { return sim_sizes_; }
    const colvec_type<uint_t>& sim_sizes() const { return sim_sizes_; }

    // This function is only valid once create_tiles() has been called.
    uint_t n_tiles(size_t gridpt_idx) const { return n_tiles_[gridpt_idx]; }
    uint_t n_tiles() const { return tiles_.size(); }
    uint_t n_gridpts() const { return thetas_.cols(); }
    uint_t n_params() const { return thetas_.rows(); }

    /*
     * Returns true if the tile specified by tile_idx
     * has ISH configuration such that null hypothesis for hypo_idx is true.
     * Note that this function is for non-regular tiles.
     * This function is only valid once create_tiles() has been called.
     */
    bool check_null(
            size_t tile_idx,
            size_t hypo_idx) const
    { 
        return tiles_[tile_idx].check_null(hypo_idx); 
    }

    /*
     * Returns true if the tile tile_idx
     * is a regular tile, i.e. a rectangular tile.
     * This function is only valid once create_tiles() has been called.
     */
    bool is_regular(size_t tile_idx) const 
    { 
        return tiles_[tile_idx].is_regular(); 
    }

    /*
     * Returns the vector of tiles.
     */
    const auto& tiles() const { return tiles_; }

private:

    mat_type<value_t> thetas_;      // matrix of theta vectors
    mat_type<value_t> radii_;       // matrix of radius vectors
    colvec_type<uint_t> sim_sizes_; // vector of simulation sizes
    colvec_type<uint_t> n_tiles_;   // n_tiles_[i] = number of tiles for ith gridpoint

    std::vector<tile_t> tiles_;     // vector of tiles (flattened across all gridpoints)
};

} // namespace kevlar
