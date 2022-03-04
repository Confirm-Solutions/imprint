#include <random>
#include <Eigen/Core>

template <class GenType, class RNGType>
void unif_sampler(GenType& gen, RNGType& rng, size_t m, size_t n)
{
    std::uniform_real_distribution<double> unif(0., 1.);
    rng = Eigen::MatrixXd::NullaryExpr(m, n, 
            [&](auto, auto) { return unif(gen); });
};
    
