#include <imprint_bits/grid/grid_range.hpp>
#include <imprint_bits/grid/gridder.hpp>
#include <imprint_bits/grid/hyperplane.hpp>
#include <imprint_bits/grid/tile.hpp>
#include <imprint_bits/model/binomial/simple_selection.hpp>
#include <testutil/base_fixture.hpp>
#include <testutil/model/binomial/simple_selection.hpp>

namespace imprint {
namespace model {
namespace binomial {

/*
 * To keep consistent with previous implementation,
 * the hyperplane object must identify whether
 * the tile is oriented w.r.t. the hyperplane if and only if
 * the center is in the positive orientation.
 */
struct MockHyperPlane : grid::HyperPlane<double> {
    using base_t = grid::HyperPlane<double>;
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
        theta_1d = grid::Gridder::make_grid(n_thetas, -1., 1.);
        radius = grid::Gridder::radius(n_thetas, -1., 1.);

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
    using tile_t = grid::Tile<value_t>;
    using gen_t = std::mt19937;
    using hp_t = MockHyperPlane;
    using gr_t = grid::GridRange<value_t, uint_t, tile_t>;
    using bckt_legacy_t = legacy::BinomialControlkTreatment;
    using bckt_t = SimpleSelection<value_t>;

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

struct bckt_state_fixture : bckt_fixture {
   protected:
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
    bckt_t b_new(n_arms, n_samples, ph2_size, thresholds);
    bckt_legacy_t b_leg(n_arms, ph2_size, n_samples, prob_1d, prob_endpt_1d,
                        hypos);

    auto sgs =
        b_new.make_sim_global_state<gen_t, value_t, uint_t, gr_t>(grid_range);

    auto s_new = sgs.make_sim_state(seed);
    state_leg_t s_leg(b_leg);

    // get legacy rejections
    state_gen(s_leg);
    dAryInt bits(n_thetas, n_arms);
    colvec_type<uint_t> expected(bits.n_unique());
    for (int i = 0; i < expected.size(); ++i, ++bits) {
        expected[i] = (s_leg.test_stat(bits) > threshold);
    }

    // get new rejections
    colvec_type<uint_t> actual(grid_range.n_tiles());
    s_new->simulate(actual);

    expect_eq_vec(actual, expected);
}

}  // namespace binomial
}  // namespace model
}  // namespace imprint
