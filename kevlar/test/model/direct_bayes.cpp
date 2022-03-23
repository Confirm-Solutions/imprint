#include <gtest/gtest.h>

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
namespace {

using mat_t = DirectBayesBinomialControlkTreatment<>::mat_t;
using vec_t = DirectBayesBinomialControlkTreatment<>::vec_t;

struct MockHyperPlane : HyperPlane<double> {
    using base_t = HyperPlane<double>;
    using base_t::base_t;
};
using value_t = double;
using uint_t = uint32_t;
using tile_t = Tile<value_t>;
using hp_t = MockHyperPlane;
using gr_t = GridRange<value_t, uint_t, tile_t>;
using bckt_t = DirectBayesBinomialControlkTreatment<value_t>;

const double mu_sig_sq = 100;
const double alpha_prior = 0.0005;
const double beta_prior = 0.000005;
const int n_points = 50;
const int n_arm_size = 15;
const auto tol = 1e-8;
const size_t n_arms = 2;
const size_t n_samples = 250;
const value_t threshold = 0.3;
const size_t n_thetas = 10;

vec_t get_thresholds() {
    vec_t thresholds(1);
    thresholds.fill(threshold);
    return thresholds;
}

gr_t get_grid_range() {
    using value_t = double;
    using uint_t = uint32_t;
    using tile_t = Tile<value_t>;
    using hp_t = MockHyperPlane;
    auto theta_1d = Gridder::make_grid(n_thetas, -1., 1.);
    auto radius = Gridder::radius(n_thetas, -1., 1.);

    colvec_type<value_t> normal(n_arms);
    normal << 1, -1;  // H_0: p[1] <= p[0]
    std::vector<hp_t> hps;
    hps.emplace_back(normal, 0);

    // only thetas and radii need to be populated.

    // populate theta as the cartesian product of theta_1d
    gr_t grid_range(n_arms, ipow(n_thetas, n_arms));
    auto& thetas = grid_range.thetas();
    dAryInt bits(n_thetas, n_arms);
    for (size_t j = 0; j < grid_range.n_gridpts(); ++j) {
        for (size_t i = 0; i < n_arms; ++i) {
            thetas(i, j) = theta_1d[bits()[i]];
        }
        ++bits;
    }

    // populate radii as fixed radius
    grid_range.radii().array() = radius;

    // create tile information
    grid_range.create_tiles(hps);

    EXPECT_EQ(grid_range.n_tiles(0), 2);
    EXPECT_EQ(grid_range.n_tiles(1), 1);
    EXPECT_EQ(grid_range.n_tiles(2), 1);
    EXPECT_EQ(grid_range.n_tiles(3), 1);
    return grid_range;
}

DirectBayesBinomialControlkTreatment<> get_test_class() {
    bckt_t b_new(n_arms, n_samples, get_thresholds());
    return b_new;
}

TEST_F(base_fixture, TestConditionalExceedProbGivenSigma) {
    for (bool use_fast : {true, false}) {
        mat_t got = DirectBayesBinomialControlkTreatment<>::
            conditional_exceed_prob_given_sigma(
                1.10517092, 0.1, Eigen::Vector4d{12.32, 10.08, 11.22, 10.08},
                Eigen::Vector4d{0.24116206, -0.94446161, 0.66329422,
                                0.94446161},
                Eigen::Vector<value_t, 1>{-0.40546511},
                Eigen::Vector4d{0, 0, 0, 0}, use_fast);
        Eigen::Vector4d want;
        want << 0.9892854091921082, 0.0656701203047288, 0.999810960134644,
            0.9999877861068269;
        EXPECT_TRUE(got.isApprox(want, tol));
        got = DirectBayesBinomialControlkTreatment<>::
            conditional_exceed_prob_given_sigma(
                1.01445965e-8, 0.1, Eigen::Vector4d{12.32, 10.08, 11.22, 10.08},
                Eigen::Vector4d{0.24116206, -0.94446161, 0.66329422,
                                0.94446161},
                Eigen::Vector<value_t, 1>{-0.40546511},
                Eigen::Vector4d{0, 0, 0, 0}, use_fast);
        want << 0.9999943915784785, 0.999994391552775, 0.9999943915861994,
            0.9999943915892988;
        EXPECT_TRUE(got.isApprox(want, tol));
    }
};

TEST_F(base_fixture, TestGetFalseRejections) {
    const auto [quadrature_points, weighted_density_logspace] =
        DirectBayesBinomialControlkTreatment<>::get_quadrature(
            alpha_prior, beta_prior, n_points, n_arm_size);
    vec_t phat = Eigen::Vector4d{3, 8, 5, 4};
    phat.array() /= 15;
    Eigen::Vector4d want{0.64462095, 0.80224266, 0.71778699, 0.67847136};
    auto thresholds = get_thresholds();
    auto got = DirectBayesBinomialControlkTreatment<>::get_false_rejections(
        phat, quadrature_points, weighted_density_logspace, thresholds,
        n_arm_size, mu_sig_sq, true);
    EXPECT_TRUE(got.isApprox(want, tol));
    got = DirectBayesBinomialControlkTreatment<>::get_false_rejections(
        phat, quadrature_points, weighted_density_logspace, thresholds,
        n_arm_size, mu_sig_sq, false);
    EXPECT_TRUE(got.isApprox(want, tol));
};

TEST_F(base_fixture, TestFasterInvert) {
    auto v = Eigen::Vector4d{1, 2, 3, 4};
    double d = 0.5;
    const auto got = DirectBayesBinomialControlkTreatment<>::faster_invert(
        1. / v.array(), d);
    mat_t m = v.asDiagonal();
    m.array() += d;
    mat_t want = m.inverse();
    EXPECT_TRUE(want.isApprox(got, tol));
};

TEST_F(base_fixture, GetGridRange) { get_grid_range(); };

TEST_F(base_fixture, TestRejLen) {
    PRINT("construct");
    auto model = get_test_class();
    auto grid_range = get_grid_range();
    model.set_grid_range(grid_range);
    auto state = model.make_state();
    size_t seed = 3214;
    std::mt19937 gen;
    gen.seed(seed);
    state->gen_rng(gen);
    PRINT("gen_suff");
    state->gen_suff_stat();
    PRINT("rej_len");
    colvec_type<uint32_t>
        rej_len;  // number of models that rejects for each gridpt
    state->rej_len(rej_len);
    // EXPECT_TRUE(false);
};
}  // namespace
}  // namespace kevlar