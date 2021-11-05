#pragma once
#include <Eigen/Core>

namespace kevlar {

template <class ValueType>
struct RunningMean
{
    using value_t = ValueType;

    template <class MatType>
    void update(const MatType& m)
    {
        value_t new_factor = 1/static_cast<value_t>(n_+1);
        mean_.array() = (1-new_factor) * mean_.array() + new_factor * m.array();
        ++n_;
    }

    void reset(size_t m, size_t n)
    {
        mean_.setZero(m, n);
        n_ = 0;
    }

    auto& operator()() { return mean_; }
    const auto& operator()() const { return mean_; }

private:
    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;
    mat_t mean_;
    unsigned int n_ = 0;
};

} // namespace kevlar
