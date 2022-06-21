#pragma once
#include <bitset>
#include <imprint_bits/grid/decl.hpp>
#include <imprint_bits/util/d_ary_int.hpp>
#include <imprint_bits/util/macros.hpp>
#include <imprint_bits/util/types.hpp>

namespace imprint {
namespace grid {

/*
 * This class represents a tile associated with a gridpoint.
 * It is the region on which we will compute the upper-bound estimates
 * (supremum) and the region associated with an intersection hypothesis space.
 */
template <class ValueType>
struct Tile {
    using value_t = ValueType;

   private:
    std::vector<colvec_type<value_t>> vertices_;  // vertices of the actual tile
    Eigen::Map<const colvec_type<value_t>> center_;  // center of tile
    Eigen::Map<const colvec_type<value_t>>
        radius_;  // radius that defines the bounds
                  // of the tile centered at center_

   public:
    struct FullVertexIterator {
        using difference_type = std::ptrdiff_t;
        using value_type = colvec_type<value_t>;
        using pointer = const value_type*;
        using reference = const value_type&;
        using iterator_category = std::forward_iterator_tag;

       private:
        const std::reference_wrapper<const Tile> outer_ref_;
        dAryInt bits_;
        colvec_type<value_t> v_;
        size_t cnt_;

       public:
        FullVertexIterator(const Tile& outer, size_t cnt)
            : outer_ref_{outer},
              bits_(2, outer.n_params()),
              v_(outer.n_params()),
              cnt_(cnt) {
            if (cnt_ < bits_.n_unique()) {
                for (size_t i = 0; i < cnt_; ++i, ++bits_)
                    ;
                auto&& dbits = bits_().template cast<value_t>();
                auto&& dir = (2 * dbits.array() - 1).matrix();
                v_ = outer_ref_.get().regular_vertex(dir);
            }
        }

        FullVertexIterator& operator++() {
            ++cnt_;
            ++bits_;
            auto&& dbits = bits_().template cast<value_t>();
            auto&& dir = (2 * dbits.array() - 1).matrix();
            v_ = outer_ref_.get().regular_vertex(dir);
            return *this;
        }
        IMPRINT_STRONG_INLINE reference operator*() const { return v_; }
        IMPRINT_STRONG_INLINE pointer operator->() const { return &v_; }

        IMPRINT_STRONG_INLINE
        constexpr bool operator==(const FullVertexIterator& it2) const {
            return (this->cnt_ == it2.cnt_) &&
                   (&this->outer_ref_.get() == &it2.outer_ref_.get());
        }

        IMPRINT_STRONG_INLINE
        constexpr bool operator!=(const FullVertexIterator& it2) const {
            return (this->cnt_ != it2.cnt_) ||
                   (&this->outer_ref_.get() != &it2.outer_ref_.get());
        }

        IMPRINT_STRONG_INLINE const auto& bits() { return bits_; }
    };

    Tile() : vertices_(), center_(nullptr, 0), radius_(nullptr, 0) {}

    Tile(const Eigen::Ref<const colvec_type<value_t>>& center,
         const Eigen::Ref<const colvec_type<value_t>>& radius)
        : vertices_(),
          center_(center.data(), center.size()),
          radius_(radius.data(), radius.size()) {}

    Tile(const Tile& t)
        : vertices_(t.vertices_),
          center_(t.center_.data(), t.center_.size()),
          radius_(t.radius_.data(), t.radius_.size()) {}
    Tile(Tile&& t)
        : vertices_(std::move(t.vertices_)),
          center_(t.center_.data(), t.center_.size()),
          radius_(t.radius_.data(), t.radius_.size()) {}
    Tile& operator=(const Tile& t) {
        vertices_ = t.vertices_;
        new (&center_) Eigen::Map<const colvec_type<value_t>>(t.center_.data(),
                                                              t.center_.size());
        new (&radius_) Eigen::Map<const colvec_type<value_t>>(t.radius_.data(),
                                                              t.radius_.size());
        return *this;
    }
    Tile& operator=(Tile&& t) {
        vertices_ = std::move(t.vertices_);
        new (&center_) Eigen::Map<const colvec_type<value_t>>(t.center_.data(),
                                                              t.center_.size());
        new (&radius_) Eigen::Map<const colvec_type<value_t>>(t.radius_.data(),
                                                              t.radius_.size());
        return *this;
    }

    /*
     * Appends a new vertex object initialized with v.
     * Note that populating the vertices automatically converts
     * this tile to be non-regular.
     * User must call make_regular(),
     * or equivalently, resize the vertices matrix to be empty,
     * to make the tile regular again.
     */
    template <class VecType>
    IMPRINT_STRONG_INLINE void emplace_back(VecType&& v) {
        vertices_.emplace_back(std::forward<VecType>(v));
    }

    /*
     * Return iterators iterating through the vertices.
     */
    IMPRINT_STRONG_INLINE auto begin() { return vertices_.begin(); }
    IMPRINT_STRONG_INLINE auto end() { return vertices_.end(); }
    IMPRINT_STRONG_INLINE auto begin() const { return vertices_.begin(); }
    IMPRINT_STRONG_INLINE auto end() const { return vertices_.end(); }

    /*
     * Return iterators iterating through the vertices
     * of the full rectangular tile defined by the center and radius.
     */
    IMPRINT_STRONG_INLINE auto begin_full() const {
        return FullVertexIterator(*this, 0);
    }
    IMPRINT_STRONG_INLINE auto end_full() const {
        return FullVertexIterator(*this, ipow(2, n_params()));
    }

    IMPRINT_STRONG_INLINE auto n_params() const { return center_.size(); }
    IMPRINT_STRONG_INLINE auto center() const { return center_; }
    IMPRINT_STRONG_INLINE auto radius() const { return radius_; }
    template <class C>
    IMPRINT_STRONG_INLINE void center(const C& c) {
        new (&center_)
            Eigen::Map<const colvec_type<value_t>>(c.data(), c.size());
    }
    template <class R>
    IMPRINT_STRONG_INLINE void radius(const R& r) {
        new (&radius_)
            Eigen::Map<const colvec_type<value_t>>(r.data(), r.size());
    }

    IMPRINT_STRONG_INLINE void make_regular() { vertices_.clear(); }
    IMPRINT_STRONG_INLINE void clear() { vertices_.clear(); }
    IMPRINT_STRONG_INLINE bool is_regular() const {
        return (vertices_.size() == 0);
    }

    /*
     * Computes a regular tile vertex based on
     * the direction to take radius.
     *
     * @param   b   vector of -1,1's where
     *              b[i] is the direction bit for ith axis.
     *              Assumed to have same dimensions as center
     *              and radius.
     */
    template <class BitsType>
    IMPRINT_STRONG_INLINE auto regular_vertex(const BitsType& b) const {
        return center_ + b.cwiseProduct(radius_);
    }

    // Helper functions for pickling
    IMPRINT_STRONG_INLINE auto& vertices__() { return vertices_; }
    IMPRINT_STRONG_INLINE const auto& vertices__() const { return vertices_; }
};

}  // namespace grid
}  // namespace imprint
