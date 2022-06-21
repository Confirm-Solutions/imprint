#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <imprint_bits/grid/grid_range.hpp>
#include <imprint_bits/grid/gridder.hpp>
#include <imprint_bits/grid/hyperplane.hpp>
#include <imprint_bits/grid/tile.hpp>
#include <imprint_bits/model/binomial/direct_bayes.hpp>
#include <imprint_bits/util/macros.hpp>
#include <iostream>
#include <testutil/base_fixture.hpp>

namespace imprint {
namespace model {
namespace binomial {
namespace {

template <class ValueType>
struct MockHyperPlane : grid::HyperPlane<ValueType> {
    using base_t = grid::HyperPlane<ValueType>;
    using base_t::base_t;
};

struct direct_bayes_fixture : base_fixture {
   protected:
    using gen_t = std::mt19937;
    using value_t = double;
    using uint_t = uint32_t;
    using tile_t = grid::Tile<value_t>;
    using hp_t = MockHyperPlane<value_t>;
    using gr_t = grid::GridRange<value_t, uint_t, tile_t>;

    using model_t = DirectBayes<value_t>;
    using sgs_t =
        typename model_t::SimGlobalState<gen_t, value_t, uint_t, gr_t>;
    using ss_t = typename sgs_t::sim_state_t;

    using mat_t = mat_type<value_t>;
    using vec_t = colvec_type<value_t>;

    const value_t mu_sig_sq = 100;
    const value_t alpha_prior = 0.0005;
    const value_t beta_prior = 0.000005;
    const value_t efficacy_threshold = 0.3;
    const int n_integration_points = 50;
    const int n_arm_size = 15;
    const value_t tol = 1e-8;
    const size_t n_arms = 4;
    const colvec_type<value_t, 1> critical_values{0.95};
    const size_t n_thetas = 4;

    vec_t get_efficacy_thresholds(int n) const {
        Eigen::Vector<value_t, Eigen::Dynamic> efficacy_thresholds(n);
        efficacy_thresholds.fill(efficacy_threshold);
        return efficacy_thresholds;
    }

    gr_t get_grid_range() const {
        auto theta_1d = grid::Gridder::make_grid(n_thetas, -1., 1.);
        auto radius = grid::Gridder::radius(n_thetas, -1., 1.);

        colvec_type<value_t> normal(n_arms);
        std::vector<hp_t> hps;
        for (size_t i = 0; i < n_arms; ++i) {
            normal.setZero();
            normal(i) = -1;
            hps.emplace_back(normal, logit(efficacy_threshold));
        }

        // populate theta as the cartesian product of theta_1d
        dAryInt bits(n_thetas, n_arms);
        gr_t grid_range(n_arms, bits.n_unique());
        auto& thetas = grid_range.thetas();
        for (size_t j = 0; j < grid_range.n_gridpts(); ++j, ++bits) {
            for (size_t i = 0; i < n_arms; ++i) {
                thetas(i, j) = theta_1d[bits()[i]];
            }
        }

        // populate radii as fixed radius
        grid_range.radii().array() = radius;

        // create tile information
        grid_range.create_tiles(hps);
        grid_range.prune();

        return grid_range;
    }

    model_t get_test_class() const {
        model_t b_new(n_arms, n_arm_size, critical_values,
                      get_efficacy_thresholds(n_arms));
        return b_new;
    }
};

TEST_F(direct_bayes_fixture, TestConditionalExceedProbGivenSigma) {
    Eigen::Vector4d logit_efficacy_thresholds;
    logit_efficacy_thresholds.fill(-0.40546511);
    for (bool use_fast : {true, false}) {
        vec_t got = sgs_t::conditional_exceed_prob_given_sigma(
            1.10517092, 0.1, Eigen::Vector4d{12.32, 10.08, 11.22, 10.08},
            Eigen::Vector4d{0.24116206, -0.94446161, 0.66329422, 0.94446161},
            logit_efficacy_thresholds, Eigen::Vector4d{0, 0, 0, 0}, use_fast);
        Eigen::Vector4d want;
        want << 0.9892854091921082, 0.0656701203047288, 0.999810960134644,
            0.9999877861068269;
        expect_near_vec(got, want, tol);
        got = sgs_t::conditional_exceed_prob_given_sigma(
            1.01445965e-8, 0.1, Eigen::Vector4d{12.32, 10.08, 11.22, 10.08},
            Eigen::Vector4d{0.24116206, -0.94446161, 0.66329422, 0.94446161},
            logit_efficacy_thresholds, Eigen::Vector4d{0, 0, 0, 0}, use_fast);
        want << 0.9999943915784785, 0.999994391552775, 0.9999943915861994,
            0.9999943915892988;
        expect_near_vec(got, want, tol);
    }
};

TEST_F(direct_bayes_fixture, TestGetPosteriorExcedanceProbs) {
    const auto [quadrature_points, weighted_density_logspace] =
        sgs_t::get_quadrature(alpha_prior, beta_prior, n_integration_points,
                              n_arm_size);
    vec_t phat = Eigen::Vector4d{3, 8, 5, 4};
    phat.array() /= 15;
    Eigen::Vector<value_t, 4> want{0.64462095, 0.80224266, 0.71778699,
                                   0.67847136};
    for (bool use_optimized : {true, false}) {
        auto got = sgs_t::get_posterior_exceedance_probs(
            phat, quadrature_points, weighted_density_logspace,
            get_efficacy_thresholds(4), n_arm_size, mu_sig_sq, use_optimized);
        expect_near_vec(got, want, tol);
    }
};

TEST_F(direct_bayes_fixture, TestFasterInvert) {
    auto v = Eigen::Vector4d{1, 2, 3, 4};
    double d = 0.5;
    const auto got = sgs_t::faster_invert(1. / v.array(), d);
    mat_t m = v.asDiagonal();
    m.array() += d;
    mat_t want = m.inverse();
    expect_near_mat(got, want, tol);
};

TEST_F(direct_bayes_fixture, GetGridRange) {
    auto grid_range = get_grid_range();
    EXPECT_EQ(grid_range.n_tiles(0), 1);
    EXPECT_EQ(grid_range.n_tiles(1), 1);
    EXPECT_EQ(grid_range.n_tiles(2), 1);
    EXPECT_EQ(grid_range.n_tiles(3), 2);
};

TEST_F(direct_bayes_fixture, TestRejLen) {
    size_t seed = 3214;
    auto model = get_test_class();
    auto grid_range = get_grid_range();
    auto sgs = model.make_sim_global_state<gen_t, value_t, uint_t>(grid_range);
    auto state = sgs.make_sim_state(seed);
    colvec_type<uint_t> actual(grid_range.n_tiles());
    state->simulate(actual);
    colvec_type<uint_t> expected(grid_range.n_tiles());
    expected << 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1,
        1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0,
        0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1,
        1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1,
        0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1,
        1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0,
        1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1;
    expect_eq_vec(actual, expected);
};
}  // namespace
}  // namespace binomial
}  // namespace model
}  // namespace imprint
