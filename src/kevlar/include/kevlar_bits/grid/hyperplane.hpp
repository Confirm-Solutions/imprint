#pragma once
#include <kevlar_bits/util/types.hpp>

namespace kevlar {

template <class ValueType>
struct HyperPlane
{
    using value_t = ValueType;

    HyperPlane(
            const Eigen::Ref<const colvec_type<value_t>>& normal,
            value_t shift)
        : normal_(normal.data(), normal.size())
        , shift_(shift)
    {}

    /*
     * Finds the orientation of a vector v w.r.t. 
     * the current hyperplane object.
     * Returns one of neg, on, pos if v is in the
     * negative, boundary, or positive side of hyperplane,
     * respectively.
     */
    template <class VecType>
    orient_type find_orient(const VecType& v) const
    {
        value_t ctv = normal_.dot(v);
        bool is_neg = (ctv < shift_);
        bool is_on = (ctv == shift_);
        return (is_neg) ? orient_type::neg :
            ((is_on) ? orient_type::on : orient_type::pos);
    }

    /*
     * Finds the directional weight alpha to get 
     * the intersected point: v + alpha * d.
     * Returns alpha in (0,1) if successful,
     * otherwise returns 0.
     */
    template <class VType, class DType>
    value_t intersect(const VType& v, const DType& d) const
    {
        auto ntd = normal_.dot(d);
        if (ntd == 0) return 0;
        auto ntv = normal_.dot(v);
        return (ntv - shift_)/ntd + 1.;
    }

private:
    const Eigen::Map<const colvec_type<value_t> > normal_; // normal vector to hyperplane
    const value_t shift_; // affine shift
};

} // namespace kevlar
