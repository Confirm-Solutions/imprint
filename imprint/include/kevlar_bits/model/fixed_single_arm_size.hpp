#pragma once
#include <cstddef>

namespace imprint {
namespace model {

/*
 * Base class with each arm having a fixed, common arm size.
 */
struct FixedSingleArmSize {
   private:
    size_t n_arms_;
    size_t n_arm_samples_;

   public:
    constexpr size_t n_arms() const { return n_arms_; }
    constexpr size_t n_arm_samples() const { return n_arm_samples_; }

    constexpr FixedSingleArmSize(size_t n_arms, size_t n_arm_samples)
        : n_arms_(n_arms), n_arm_samples_(n_arm_samples) {}
};

}  // namespace model
}  // namespace imprint
