#pragma once
#include <cstddef>
#include <random>
#include <Eigen/Core>

namespace kevlar {

struct ControlkTreatmentBase
{
    ControlkTreatmentBase(
            size_t n_arms,
            size_t ph2_size,
            size_t n_samples
            )
        : n_arms_(n_arms)
        , ph2_size_(ph2_size)
        , n_samples_(n_samples)
    {}

    size_t n_samples() const { return n_samples_; }
    size_t n_arms() const { return n_arms_; }

    /* Helper static interface */
    template <class GenType, class OutType>
    static void uniform(double min, double max, GenType&& gen, OutType&& out, size_t m, size_t n) {
        std::uniform_real_distribution<double> unif(min, max);
        out = Eigen::MatrixXd::NullaryExpr(m, n, 
                [&](auto, auto) { return unif(gen); });
    }

protected:
    size_t n_arms_;
    size_t ph2_size_;
    size_t n_samples_;
};

} // namespace kevlar
