#include <imprint_bits/grid/grid_range.hpp>
#include <imprint_bits/grid/tile.hpp>
#include <imprint_bits/model/binomial/common/fixed_n_default.hpp>
#include <imprint_bits/util/math.hpp>
#include <testutil/base_fixture.hpp>

namespace imprint {
namespace model {
namespace binomial {

/*
 * Wrap the class to dummy-implement the virtual functions.
 */
template <class GenType, class ValueType, class UIntType, class GridRangeType>
struct SimGlobalStateFixedNDefaultWrap
    : SimGlobalStateFixedNDefault<GenType, ValueType, UIntType, GridRangeType> {
    struct SimState;

    using base_t = SimGlobalStateFixedNDefault<GenType, ValueType, UIntType,
                                               GridRangeType>;
    using typename base_t::gen_t;
    using typename base_t::interface_t;
    using typename base_t::uint_t;
    using sim_state_t = SimState;

    using base_t::base_t;

    std::unique_ptr<typename interface_t::sim_state_t> make_sim_state(
        size_t seed) const override {
        return std::make_unique<sim_state_t>(*this, seed);
    }
};

template <class GenType, class ValueType, class UIntType, class GridRangeType>
struct SimGlobalStateFixedNDefaultWrap<GenType, ValueType, UIntType,
                                       GridRangeType>::SimState
    : base_t::sim_state_t {
    using outer_t = SimGlobalStateFixedNDefaultWrap;
    using base_t = typename outer_t::base_t::sim_state_t;

    using base_t::base_t;
    void simulate(Eigen::Ref<colvec_type<uint_t>>) override{};
};

// ======================================================
// TEST SimGlobalState
// ======================================================

struct sgs_fixed_n_default_fixture : base_fixture {
   protected:
    using gen_t = std::mt19937;
    using value_t = double;
    using uint_t = uint32_t;
    using tile_t = grid::Tile<value_t>;
    using gr_t = grid::GridRange<value_t, uint_t, tile_t>;
    using sgs_t = SimGlobalStateFixedNDefaultWrap<gen_t, value_t, uint_t, gr_t>;
};

TEST_F(sgs_fixed_n_default_fixture, one_arm) {
    size_t d = 1;  // number of params
    size_t n = 5;  // number of gridpts
    size_t n_arm_samples = 2;
    // number of arm size;
    // not important for this test

    gr_t gr(d, n);
    auto& thetas = gr.thetas();
    thetas.row(0) << -0.5, 0., -0.5, 0.3, .3;

    sgs_t sgs(n_arm_samples, gr);

    // test the unique probs
    colvec_type<double> expected_pu(3);
    expected_pu << -0.5, 0., 0.3;
    expected_pu.array() = sigmoid(expected_pu.array());
    auto& pu = sgs.probs_unique_arm(0);
    expect_double_eq_vec(pu, expected_pu);

    // test the bits
    colvec_type<uint_t> expected_bits(n);
    expected_bits << 0, 1, 0, 2, 2;
    auto bits = sgs.bits().row(0);
    expect_eq_vec(bits, expected_bits);

    // test the stride
    EXPECT_EQ(sgs.stride(0), 0);
    EXPECT_EQ(sgs.stride(1), expected_pu.size());
}

TEST_F(sgs_fixed_n_default_fixture, two_arms) {
    size_t d = 2;  // number of params
    size_t n = 5;  // number of gridpts
    size_t n_arm_samples = 2;
    // number of arm size;
    // not important for this test

    gr_t gr(d, n);
    auto& thetas = gr.thetas();
    thetas.row(0) << -0.5, 0., -0.5, 0.3, .3;
    thetas.row(1) << 0.1, -0.3, 0.1, -0.3, 0.2;

    sgs_t sgs(n_arm_samples, gr);

    //// Arm 1:
    {
        // test the unique probs
        colvec_type<double> expected_pu(3);
        expected_pu << -0.5, 0., 0.3;
        expected_pu.array() = sigmoid(expected_pu.array());
        auto& pu = sgs.probs_unique_arm(0);
        expect_double_eq_vec(pu, expected_pu);

        // test the bits
        colvec_type<uint_t> expected_bits(n);
        expected_bits << 0, 1, 0, 2, 2;
        auto bits = sgs.bits().row(0);
        expect_eq_vec(bits, expected_bits);

        // test the stride
        EXPECT_EQ(sgs.stride(0), 0);
        EXPECT_EQ(sgs.stride(1), expected_pu.size());
    }

    //// Arm 2:
    {
        // test the unique probs
        colvec_type<double> expected_pu(3);
        expected_pu << -0.3, 0.1, 0.2;
        expected_pu.array() = sigmoid(expected_pu.array());
        auto& pu = sgs.probs_unique_arm(1);
        expect_double_eq_vec(pu, expected_pu);

        // test the bits
        colvec_type<uint_t> expected_bits(n);
        expected_bits << 1, 0, 1, 0, 2;
        auto bits = sgs.bits().row(1);
        expect_eq_vec(bits, expected_bits);

        // test the stride
        EXPECT_EQ(sgs.stride(2), sgs.stride(1) + expected_pu.size());
    }
}

// ======================================================
// TEST SimState
// ======================================================

struct ss_fixed_n_default_fixture : sgs_fixed_n_default_fixture {
    void SetUp() override { gen.seed(0); }

   protected:
    using ss_t = typename sgs_t::sim_state_t;
    gen_t gen;
};

TEST_F(ss_fixed_n_default_fixture, two_arm_suff_stat_score) {
    size_t d = 2;  // number of params
    size_t n = 5;  // number of gridpts
    size_t n_arm_samples = 2;
    // number of arm size;
    // IS important for this test

    gr_t gr(d, n);
    auto& thetas = gr.thetas();
    thetas.row(0) << -0.5, 0., -0.5, 0.3, .3;
    thetas.row(1) << 0.1, -0.3, 0.1, -0.3, 0.2;

    sgs_t sgs(n_arm_samples, gr);
    ss_t ss = *static_cast<ss_t*>(sgs.make_sim_state(0).get());

    ss.generate_data();
    ss.generate_sufficient_stats();

    std::vector<colvec_type<double>> pu_v(2);
    pu_v[0].resize(3);
    pu_v[1].resize(3);
    pu_v[0] << -0.5, 0., 0.3;
    pu_v[1] << -0.3, 0.1, 0.2;
    pu_v[0].array() = sigmoid(pu_v[0].array());
    pu_v[1].array() = sigmoid(pu_v[1].array());

    // test sufficient stats
    std::vector<colvec_type<double>> ss_expected(2);
    ss_expected[0].resize(3);
    ss_expected[1].resize(3);
    {
        for (size_t i = 0; i < pu_v.size(); ++i) {
            auto unifs_i = ss.uniform_randoms().col(i);
            auto& pu = pu_v[i];
            for (int j = 0; j < pu.size(); ++j) {
                auto actual = (unifs_i.array() < pu[j]).count();
                ss_expected[i](j) = actual;
                auto expected = ss.sufficient_stats_arm(i)(j);
                EXPECT_EQ(actual, expected);
            }
        }
    }

    // test score
    {
        mat_type<uint_t> bits(d, n);
        bits << 0, 1, 0, 2, 2, 1, 0, 1, 0, 2;
        mat_type<double> expected;
        expected = expected.NullaryExpr(d, n, [&](auto i, auto j) {
            return ss_expected[i][bits(i, j)] -
                   n_arm_samples * pu_v[i][bits(i, j)];
        });

        colvec_type<double> out(d);
        for (size_t i = 0; i < gr.n_gridpts(); ++i) {
            ss.score(i, out);
            auto expected_i = expected.col(i);
            expect_double_eq_vec(out, expected_i);
        }
    }
}

// ======================================================
// TEST Imprint Bound State
// ======================================================

struct kbs_fixed_n_default_fixture : base_fixture {
   protected:
    using value_t = double;
    using uint_t = uint32_t;
    using tile_t = grid::Tile<value_t>;
    using gr_t = grid::GridRange<value_t, uint_t, tile_t>;
    using kbs_t = ImprintBoundStateFixedNDefault<gr_t>;
};

TEST_F(kbs_fixed_n_default_fixture, apply_eta_jacobian) {
    size_t d = 5;              // arbitrary
    size_t n_arm_samples = 3;  // arbitrary
    colvec_type<value_t> v;
    v.setRandom(d);
    gr_t gr(d, 1);
    gr.thetas() = v;  // dummy
    kbs_t kbs(n_arm_samples, gr);
    colvec_type<value_t> out(v.size());
    kbs.apply_eta_jacobian(0, v, out);
    expect_double_eq_vec(out, v);
}

TEST_F(kbs_fixed_n_default_fixture, covar_quadform) {
    size_t d = 3;  // number of params
    size_t n_arm_samples = 100;

    // the invariance is that
    // the values of theta and v does not matter

    gr_t gr(d, 1);
    gr.thetas().setRandom();

    kbs_t kbs(n_arm_samples, gr);

    colvec_type<value_t> v;
    v.setRandom(d);

    colvec_type<value_t> prob = sigmoid(gr.thetas().array());
    auto prob_a = prob.array();
    auto v_a = v.array();
    value_t expected =
        (n_arm_samples * v_a.square() * prob_a * (1.0 - prob_a)).sum();

    value_t actual = kbs.covar_quadform(0, v);
    EXPECT_DOUBLE_EQ(actual, expected);
}

TEST_F(kbs_fixed_n_default_fixture, hessian_quadform_bound) {
    size_t d = 3;  // number of params
    size_t n_arm_samples = 250;

    gr_t gr(d, 1);
    gr.thetas() << -0.5, 0., 0.5;
    gr.radii().array() = 0.25;
    // technically, tiles should be initialized,
    // but it should not be used.

    kbs_t kbs(n_arm_samples, gr);

    colvec_type<value_t> v(d);
    v.setRandom();

    value_t actual = kbs.hessian_quadform_bound(0, 0, v);

    // compute the expected bound
    auto theta = gr.thetas().col(0);
    auto radius = gr.radii().col(0);
    value_t expected = 0;
    value_t p = 0;
    p = sigmoid(theta[0] + radius[0]);
    expected += n_arm_samples * p * (1 - p) * v[0] * v[0];
    expected += n_arm_samples * 0.25 * v[1] * v[1];
    p = sigmoid(theta[2] - radius[2]);
    expected += n_arm_samples * p * (1 - p) * v[2] * v[2];

    EXPECT_DOUBLE_EQ(actual, expected);
}

}  // namespace binomial
}  // namespace model
}  // namespace imprint
