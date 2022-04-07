
#include <Eigen/Dense>
#include <iostream>
#include <kevlar_bits/grid/grid_range.hpp>
#include <kevlar_bits/grid/gridder.hpp>
#include <kevlar_bits/grid/hyperplane.hpp>
#include <kevlar_bits/grid/tile.hpp>
#include <kevlar_bits/model/direct_bayes_binomial_control_k_treatment.hpp>
#include <kevlar_bits/util/macros.hpp>
#include <testutil/base_fixture.hpp>

namespace kevlar {

struct MockHyperPlane;
using value_t = double;
using mat_t = DirectBayesBinomialControlkTreatment<value_t>::mat_t;
using vec_t = DirectBayesBinomialControlkTreatment<value_t>::vec_t;
using uint_t = uint32_t;
using tile_t = Tile<value_t>;
using hp_t = MockHyperPlane;
using gr_t = GridRange<value_t, uint_t, tile_t>;
using bckt_t = DirectBayesBinomialControlkTreatment<value_t>;

const double mu_sig_sq = 100;
const double alpha_prior = 0.0005;
const double beta_prior = 0.000005;
const double efficacy_threshold = 0.3;
const int n_integration_points = 50;
const int n_arm_size = 15;
const auto tol = 1e-8;
const size_t n_arms = 2;
const size_t n_samples = 250;
const Eigen::Vector<value_t, 1> critical_values{0.95};
const size_t n_thetas = 10;

vec_t get_efficacy_thresholds(int n);

gr_t get_grid_range(int n_thetas = 10, int n_arms = 2);

DirectBayesBinomialControlkTreatment<value_t> get_test_class();
}  // namespace kevlar
