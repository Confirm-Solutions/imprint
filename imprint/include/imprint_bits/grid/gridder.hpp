#pragma once
#include <Eigen/Dense>
#include <cstddef>
#include <imprint_bits/util/types.hpp>

namespace imprint {
namespace grid {

/*
 * This class is responsible for defining routines to easily
 * create a 1-dimensional grid.
 */
struct Gridder {
    template <class ValueType>
    static ValueType radius(size_t n, ValueType lower, ValueType upper) {
        assert(n);
        return (upper - lower) / (2 * n);
    }

    template <class ValueType>
    static auto make_grid(size_t n, ValueType lower, ValueType upper) {
        using value_t = ValueType;
        using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
        auto r = radius(n, lower, upper);
        return ((2. * vec_t::LinSpaced(n, 0, n - 1).array() + 1.) * r + lower)
            .matrix();
    }

    template <class ValueType>
    static auto make_endpts(size_t n, ValueType lower, ValueType upper) {
        using value_t = ValueType;
        using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;
        auto r = radius(n, lower, upper);
        return mat_t::NullaryExpr(
            2, n, [=](auto i, auto j) { return 2 * (j + i) * r + lower; });
    }
};

}  // namespace grid
}  // namespace imprint
