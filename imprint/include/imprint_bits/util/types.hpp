#pragma once
#include <Eigen/Core>

namespace imprint {

template <class T, int _Rows = Eigen::Dynamic>
using colvec_type = Eigen::Matrix<T, _Rows, 1>;

template <class T>
using rowvec_type = Eigen::Matrix<T, 1, Eigen::Dynamic>;

template <class T, int _Rows = Eigen::Dynamic, int _Cols = Eigen::Dynamic,
          int _Options = Eigen::ColMajor>
using mat_type = Eigen::Matrix<T, _Rows, _Cols, _Options>;

/*
 * Orientation type definitions.
 * In general, we will have a notion of a curve/surface
 * that splits a given space into 3 regions: "positive", "negative", "boundary".
 * The user can define which of the 3 regions correspond to the three labels.
 * For example, the hyperplane class would associate positive side as
 * the side in the same direction as the normal vector;
 * negative side as the opposite direction as the normal vector;
 * and boundary as the hyperplane itself.
 *
 *  pos = positive side.
 *  neg = negative side.
 *  on  = boundary side.
 *  non_pos = non-positive side, i.e. either neg or on.
 *  non_neg = non-negative side, i.e. either pos or on.
 *  non_on  = non-boundary side, i.e. either neg or pos.
 *  none = no relationship with any side.
 */
enum class orient_type : unsigned char {
    non_pos = 0,
    non_neg,
    non_on,
    pos,
    neg,
    on,
    none,
    // Iterators for enum class
    // these MUST come last
    end,        // end iterator
    begin = 0,  // begin iterator
};

inline constexpr orient_type& operator++(orient_type& x) {
    x = static_cast<orient_type>(static_cast<unsigned char>(x) + 1);
    return x;
}

inline constexpr bool operator<(orient_type x, orient_type y) {
    switch (y) {
        case orient_type::non_pos:
            return (x == orient_type::neg) || (x == orient_type::on);
        case orient_type::non_neg:
            return (x == orient_type::pos) || (x == orient_type::on);
        case orient_type::non_on:
            return (x == orient_type::pos) || (x == orient_type::neg);
        default:
            return false;
    }
}

inline constexpr bool operator<=(orient_type x, orient_type y) {
    return (x == y) || (x < y);
}

}  // namespace imprint
