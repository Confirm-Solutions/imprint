#pragma once
#include <Eigen/Core>

namespace kevlar {
    
template <class T>
using colvec_type = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <class T>
using rowvec_type = Eigen::Matrix<T, 1, Eigen::Dynamic>;

template <class T, int _Options=Eigen::ColMajor>
using mat_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options>;

} // namespace kevlar
