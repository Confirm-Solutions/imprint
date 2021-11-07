#pragma once
#include <Eigen/Core>

namespace kevlar {

struct dAryInt
{
    // @param   d   base d
    // @param   k   number of bits
    dAryInt(size_t d, size_t k)
        : d_(d), bits_(k)
    {
        bits_.setZero();
    }

    dAryInt& operator++()
    {
        for (int i = bits_.size()-1; i >= 0; --i) {
            auto& b = bits_(i);
            ++b;
            if (b < d_) break;
            b = 0;
        }
        return *this;
    }

    void setZero() { bits_.setZero(); }

    const auto& operator()() const { return bits_; }

private:
    size_t d_;
    Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> bits_;
};


} // namespace kevlar
