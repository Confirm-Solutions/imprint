#include <kevlar>
#include <RcppEigen.h>

using namespace Rcpp;
using namespace kevlar;
using grid_t = Gridder<grid::Rectangular>;

// Rectangular grid radius.
// [[Rcpp::export]]
double grid_radius(size_t n, double lower, double upper) {
    return grid_t::radius(n, lower, upper);
}

// Make rectangular grid.
// [[Rcpp::export]]
Eigen::VectorXd make_grid(size_t n, double lower, double upper) {
    return grid_t::make_grid(n, lower, upper);
}

// Make end points of a rectangular grid (1-d).
// [[Rcpp::export]]
Eigen::MatrixXd make_endpts(size_t n, double lower, double upper) {
    return grid_t::make_endpts(n, lower, upper);
}
