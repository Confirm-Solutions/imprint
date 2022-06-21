#pragma once
#include <imprint_bits/util/macros.hpp>
#include <imprint_bits/util/types.hpp>

namespace imprint {
namespace grid {

template <class ValueType>
struct HyperPlaneView {
    using value_t = ValueType;

   private:
    Eigen::Map<const colvec_type<value_t>>
        normal_;            // normal vector to hyperplane
    const value_t* shift_;  // affine shift

   public:
    HyperPlaneView() : normal_(nullptr, 0), shift_(nullptr) {}

    HyperPlaneView(const Eigen::Ref<const colvec_type<value_t>>& normal,
                   const value_t& shift)
        : normal_(normal.data(), normal.size()), shift_(&shift) {}

    /*
     * Finds the orientation of a vector v w.r.t.
     * the current hyperplane object.
     * Returns one of neg, on, pos if v is in the
     * negative, boundary, or positive side of hyperplane,
     * respectively.
     */
    template <class VecType>
    IMPRINT_STRONG_INLINE orient_type find_orient(const VecType& v) const {
        value_t ctv = normal_.dot(v);
        constexpr value_t tol = 1e-16;
        auto comp = ctv - *shift_;
        if (comp <= -tol) {
            return orient_type::neg;
        } else if (comp >= tol) {
            return orient_type::pos;
        }
        return orient_type::on;
    }

    /*
     * Finds the directional weight alpha to get
     * the intersected point: v + alpha * d.
     * Returns alpha in (0,1) if successful,
     * otherwise returns 0.
     */
    template <class VType, class DType>
    IMPRINT_STRONG_INLINE value_t intersect(const VType& v,
                                            const DType& d) const {
        auto ntd = normal_.dot(d);
        if (ntd == 0) return 0;
        auto ntv = normal_.dot(v);
        return (*shift_ - ntv) / ntd;
    }

    IMPRINT_STRONG_INLINE auto normal() const { return normal_; }
    IMPRINT_STRONG_INLINE
    void normal(const Eigen::Ref<const colvec_type<value_t>>& n) {
        new (&normal_)
            Eigen::Map<const colvec_type<value_t>>(n.data(), n.size());
    }
    IMPRINT_STRONG_INLINE auto shift() const { return *shift_; }
    IMPRINT_STRONG_INLINE void shift(const value_t& s) { shift_ = &s; }
};

template <class ValueType>
struct HyperPlane : HyperPlaneView<ValueType> {
   private:
    using view_t = HyperPlaneView<ValueType>;

   public:
    using typename view_t::value_t;

   private:
    colvec_type<value_t> normal_;
    value_t shift_;

    IMPRINT_STRONG_INLINE
    void reset_view() {
        this->normal(normal_);
        this->shift(shift_);
    }

   public:
    HyperPlane(const Eigen::Ref<const colvec_type<value_t>>& normal,
               const value_t& shift)
        : view_t(), normal_(normal), shift_(shift) {
        reset_view();
    }

    HyperPlane(const HyperPlane& hp)
        : view_t(), normal_(hp.normal_), shift_(hp.shift_) {
        reset_view();
    }

    HyperPlane(HyperPlane&& hp)
        : view_t(),
          normal_(std::move(hp.normal_)),
          shift_(std::move(hp.shift_)) {
        reset_view();
    }

    HyperPlane& operator=(const HyperPlane& hp) {
        normal_ = hp.normal_;
        shift_ = hp.shift_;
        reset_view();
    }

    HyperPlane& operator=(HyperPlane&& hp) {
        normal_ = std::move(hp.normal_);
        shift_ = std::move(hp.shift_);
        reset_view();
    }
};

}  // namespace grid
}  // namespace imprint
