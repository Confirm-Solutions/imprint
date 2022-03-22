#include <testutil/base_fixture.hpp>
#include <testutil/model/binomial_control_k_treatment_legacy.hpp>
#include <kevlar_bits/model/binomial_control_k_treatment.hpp>
#include <kevlar_bits/grid/grid_range.hpp>
#include <kevlar_bits/grid/tile.hpp>
#include <kevlar_bits/grid/hyperplane.hpp>
#include <kevlar_bits/grid/gridder.hpp>
#include <random>

namespace kevlar {

/*
 * To keep consistent with previous implementation,
 * the hyperplane object must identify whether
 * the tile is oriented w.r.t. the hyperplane if and only if
 * the center is in the positive orientation.
 */
struct MockHyperPlane : HyperPlane<double> {
    using base_t = HyperPlane<double>;
    using base_t::base_t;
};

/*
 * We overload this function to be compatible with legacy version.
 * Legacy version does not distinguish tiles that are split by surfaces.
 * In particular, it assumes every tile is on one side of any surface,
 * and it is associated with the side where the center of the tile lies.
 * Hence, this is overloaded to return true no matter what.
 */
template <class TileType>
inline bool is_oriented(const TileType& tile, const MockHyperPlane& hp,
                        orient_type& ori) {
    const auto& center = tile.center();
    ori = hp.find_orient(center);
    if (ori <= orient_type::non_neg) {
        ori = orient_type::non_neg;
    } else {
        ori = orient_type::non_pos;
    }
    return true;
}

struct bckt_fixture : base_fixture {
    void SetUp() override {
        // legacy setup
        // MUST BE EVENLY SPACED TO BE COMPATIBLE WITH
        // MockHyperPlane and legacy version
        theta_1d = Gridder::make_grid(n_thetas, -1., 1.);
        radius = Gridder::radius(n_thetas, -1., 1.);

        prob_1d.array() = sigmoid(theta_1d.array());
        prob_endpt_1d.resize(2, theta_1d.size());
        prob_endpt_1d.row(0).array() = sigmoid(theta_1d.array() - radius);
        prob_endpt_1d.row(1).array() = sigmoid(theta_1d.array() + radius);

        for (size_t i = 1; i < n_arms; ++i) {
            hypos.emplace_back([&, i](const dAryInt& bits) {
                return prob_1d[bits()[i]] <= prob_1d[bits()[0]];
            });
        }

        // new setup

        colvec_type<value_t> normal(n_arms);
        normal << 1, -1;  // H_0: p[1] <= p[0]
        hps.emplace_back(normal, 0);

        // only thetas and radii need to be populated.

        // populate theta as the cartesian product of theta_1d
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

        EXPECT_EQ(grid_range.n_tiles(0), 1);
        EXPECT_EQ(grid_range.n_tiles(1), 1);
        EXPECT_EQ(grid_range.n_tiles(2), 1);
        EXPECT_EQ(grid_range.n_tiles(3), 1);
    }

   protected:
    using value_t = double;
    using uint_t = uint32_t;
    using tile_t = Tile<value_t>;
    using hp_t = MockHyperPlane;
    using gr_t = GridRange<value_t, uint_t, tile_t>;
    using bckt_legacy_t = legacy::BinomialControlkTreatment;
    using bckt_t = BinomialControlkTreatment<value_t, uint_t, gr_t>;

    // common configuration

    // configuration that may want to be parametrizations
    size_t n_arms = 2;
    size_t ph2_size = 50;
    size_t n_samples = 250;
    value_t threshold = 1.96;
    value_t radius;
    size_t n_thetas = 10;

    // configuration for legacy
    colvec_type<value_t> thresholds;
    colvec_type<value_t> theta_1d;
    colvec_type<value_t> prob_1d;
    mat_type<value_t> prob_endpt_1d;
    std::vector<std::function<bool(const dAryInt&)> > hypos;

    // configuration for new
    std::vector<hp_t> hps;
    gr_t grid_range;

    bckt_fixture() : thresholds(1), grid_range(n_arms, ipow(n_thetas, n_arms)) {
        thresholds[0] = threshold;
    }
};

TEST_F(bckt_fixture, ctor) {
    bckt_t b_new(n_arms, ph2_size, n_samples, thresholds);
    bckt_legacy_t b_leg(n_arms, ph2_size, n_samples, prob_1d, prob_endpt_1d,
                        hypos);
}

TEST_F(bckt_fixture, tr_cov_test) {
    dAryInt bits(n_thetas, n_arms);
    bckt_t b_new(n_arms, ph2_size, n_samples, thresholds);
    b_new.set_grid_range(grid_range);
    bckt_legacy_t b_leg(n_arms, ph2_size, n_samples, prob_1d, prob_endpt_1d,
                        hypos);
    for (size_t i = 0; i < ipow(n_thetas, n_arms); ++i, ++bits) {
        value_t b_new_tr_cov_i =
            b_new.cov_quad(i, Eigen::VectorXd::Ones(n_arms));
        EXPECT_DOUBLE_EQ(b_new_tr_cov_i, b_leg.tr_cov(bits));
    }
}

TEST_F(bckt_fixture, tr_max_cov_test) {
    dAryInt bits(n_thetas, n_arms);
    bckt_t b_new(n_arms, ph2_size, n_samples, thresholds);
    b_new.set_grid_range(grid_range);
    bckt_legacy_t b_leg(n_arms, ph2_size, n_samples, prob_1d, prob_endpt_1d,
                        hypos);
    for (size_t i = 0; i < ipow(n_thetas, n_arms); ++i, ++bits) {
        value_t b_new_tr_max_cov_i =
            b_new.max_cov_quad(i, Eigen::VectorXd::Ones(n_arms));
        EXPECT_DOUBLE_EQ(b_new_tr_max_cov_i, b_leg.tr_max_cov(bits));
    }
}

struct bckt_state_fixture : bckt_fixture {
   protected:
    using state_t = bckt_t::StateType;
    using state_leg_t = bckt_legacy_t::StateType;

    size_t seed = 3214;
    std::mt19937 gen;

    template <class StateType>
    void state_gen(StateType& s) {
        gen.seed(seed);
        s.gen_rng(gen);
        s.gen_suff_stat();
    }
};

TEST_F(bckt_state_fixture, test_rej) {
    bckt_t b_new(n_arms, ph2_size, n_samples, thresholds);
    bckt_legacy_t b_leg(n_arms, ph2_size, n_samples, prob_1d, prob_endpt_1d,
                        hypos);

    b_new.set_grid_range(grid_range);

    state_t s_new(b_new);
    state_leg_t s_leg(b_leg);

    state_gen(s_new);
    state_gen(s_leg);

    // get legacy rejections
    colvec_type<uint_t> expected(ipow(n_thetas, n_arms));
    dAryInt bits(n_thetas, n_arms);
    for (int i = 0; i < expected.size(); ++i, ++bits) {
        expected[i] = (s_leg.test_stat(bits) > threshold);
    }

    // get new rejections
    colvec_type<uint_t> actual(grid_range.n_tiles());
    s_new.rej_len(actual);

    expect_eq_vec(actual, expected);
}

TEST_F(bckt_state_fixture, grad_test) {
    bckt_t b_new(n_arms, ph2_size, n_samples, thresholds);
    bckt_legacy_t b_leg(n_arms, ph2_size, n_samples, prob_1d, prob_endpt_1d,
                        hypos);

    b_new.set_grid_range(grid_range);

    state_t s_new(b_new);
    state_leg_t s_leg(b_leg);

    state_gen(s_new);
    state_gen(s_leg);

    // get gradient estimates from new
    mat_type<value_t> actual(grid_range.n_tiles(), n_arms);
    for (int j = 0; j < actual.rows(); ++j) {
        for (int k = 0; k < actual.cols(); ++k) {
            actual(j, k) = s_new.grad(j, k);
        }
    }

    // get gradient estimates from legacy
    colvec_type<value_t> expected(grid_range.n_tiles() * n_arms);
    Eigen::Map<mat_type<value_t> > expected_m(expected.data(),
                                              grid_range.n_tiles(), n_arms);
    for (size_t j = 0; j < n_arms; ++j) {
        dAryInt bits(n_thetas, n_arms);
        for (size_t i = 0; i < grid_range.n_tiles(); ++i, ++bits) {
            expected_m(i, j) = s_leg.grad_lr(j, bits);
        }
    }

    expect_eq_mat(actual, expected_m);
}

}  // namespace kevlar
