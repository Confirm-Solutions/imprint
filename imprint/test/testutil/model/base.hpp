#pragma once
#include <Eigen/Core>
#include <cstddef>
#include <random>

namespace imprint {

/*
 * Base class for all control + k treatment designs.
 */
struct ControlkTreatmentBase {
    /*
     * @param   n_arms      number of arms (including control).
     * @param   ph2_size    phase II number of patients in each arm.
     * @param   n_samples   number of total patients in each arm (including
     * phase II) for phase II and phase III.
     */
    ControlkTreatmentBase(size_t n_arms, size_t ph2_size, size_t n_samples)
        : n_arms_(n_arms), ph2_size_(ph2_size), n_samples_(n_samples) {}

    constexpr size_t n_samples() const { return n_samples_; }
    constexpr size_t n_arms() const { return n_arms_; }

    /* Helper static interface */
    template <class GenType, class UnifType, class OutType>
    static void uniform(size_t m, size_t n, GenType&& gen, UnifType&& unif,
                        OutType&& out) {
        out = Eigen::MatrixXd::NullaryExpr(
            m, n, [&](auto, auto) { return unif(gen); });
    }

   protected:
    size_t n_arms_;
    size_t ph2_size_;
    size_t n_samples_;
};

}  // namespace imprint
