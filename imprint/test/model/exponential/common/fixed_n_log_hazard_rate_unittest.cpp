#include <imprint_bits/grid/grid_range.hpp>
#include <imprint_bits/grid/tile.hpp>
#include <imprint_bits/model/exponential/common/fixed_n_log_hazard_rate.hpp>
#include <testutil/base_fixture.hpp>

namespace imprint {
namespace model {
namespace exponential {

// ======================================================
// TEST SimState
// ======================================================

struct ss_fixture : base_fixture {
    void SetUp() override { gen.seed(seed); }

   protected:
    using gen_t = std::mt19937;
    using value_t = double;
    using uint_t = uint32_t;
    using tile_t = grid::Tile<value_t>;
    using gr_t = grid::GridRange<value_t, uint_t, tile_t>;

    struct SimGlobalStateWrap
        : SimGlobalStateFixedNLogHazardRate<gen_t, value_t, uint_t, gr_t> {
        using base_t =
            SimGlobalStateFixedNLogHazardRate<gen_t, value_t, uint_t, gr_t>;
        using typename base_t::interface_t;

        struct SimStateWrap : base_t::sim_state_t {
            using outer_t = SimGlobalStateWrap;
            using base_t = typename outer_t::base_t::sim_state_t;
            using base_t::base_t;
            void simulate(Eigen::Ref<colvec_type<uint_t>>) override{};
        };

        using base_t::base_t;

        std::unique_ptr<typename interface_t::sim_state_t> make_sim_state(
            size_t seed) const override {
            return std::make_unique<SimStateWrap>(*this, seed);
        }
    };

    using sgs_t = SimGlobalStateWrap;
    using ss_t = typename sgs_t::SimStateWrap;

    size_t d = 2;  // number of parameters (fixed to be 2)
    size_t seed = 123;
    gen_t gen;
};

TEST_F(ss_fixture, score_test) {
    size_t n = 2;  // number of gridpts
    size_t n_arm_samples = 132;

    gr_t gr(d, n);
    auto& thetas = gr.thetas();
    thetas.row(0) << -2, 1;
    thetas.row(1) << -2, -1;

    sgs_t sgs(n_arm_samples, gr);
    ss_t ss(sgs, seed);

    ss.generate_data();
    ss.generate_sufficient_stats();

    for (size_t i = 0; i < n; ++i) {
        auto hzrd_rate = std::exp(thetas(1, i));
        ss.update_hzrd_rate(hzrd_rate);
        colvec_type<value_t> expected(d);
        mat_type<value_t, 2, 1> lmda_inv;
        lmda_inv[0] = std::exp(-thetas(0, i));
        lmda_inv[1] = lmda_inv[0] * std::exp(-thetas(1, i));
        mat_type<value_t, 2, 1> suff_stat;
        suff_stat[0] = ss.control().sum();
        suff_stat[1] = ss.treatment().sum();
        expected = (suff_stat * lmda_inv[0] - n_arm_samples * lmda_inv);

        colvec_type<value_t> out(d);
        ss.score(i, out);

        auto tol = 2e-15 * expected.array().abs().maxCoeff();
        expect_near_vec(out, expected, tol);
    }
}

// ======================================================
// TEST ImprintBoundState
// ======================================================

struct kbs_fixture : base_fixture {
    using value_t = double;
    using uint_t = uint32_t;
    using tile_t = grid::Tile<value_t>;
    using gr_t = grid::GridRange<value_t, uint_t, tile_t>;
    using kbs_t = ImprintBoundStateFixedNLogHazardRate<gr_t>;
    const size_t d = 2;
    static constexpr value_t tol = 2e-15;
};

TEST_F(kbs_fixture, apply_eta_jacobian) {
    size_t n_arm_samples = 32;

    gr_t gr(d, 1);
    gr.thetas() << 1, -2;

    kbs_t kbs(n_arm_samples, gr);

    colvec_type<value_t> v(d);
    v.setRandom();

    colvec_type<value_t> nat = gr.thetas().array().exp();
    nat[0] = -nat[0];
    nat[1] *= nat[0];
    mat_type<value_t> deta(d, d);
    deta << nat[0], 0, nat[1], nat[1];
    colvec_type<value_t> expected = deta * v;

    colvec_type<value_t> out(d);
    kbs.apply_eta_jacobian(0, v, out);

    expect_double_eq_vec(out, expected);
}

TEST_F(kbs_fixture, covar_quadform) {
    size_t n_arm_samples = 32;

    gr_t gr(d, 1);
    gr.thetas() << 1, -2;

    kbs_t kbs(n_arm_samples, gr);

    colvec_type<value_t> v(d);
    v.setRandom();

    colvec_type<value_t> lmda = gr.thetas().array().exp();
    lmda[1] *= lmda[0];
    value_t expected =
        n_arm_samples *
        ((1. / lmda.array()).square() * v.array().square()).sum();

    value_t actual = kbs.covar_quadform(0, v);

    EXPECT_NEAR(actual, expected, expected * tol);
}

TEST_F(kbs_fixture, hessian_quadform_bound) {
    size_t n_arm_samples = 32;

    gr_t gr(d, 1);

    // these values should not matter
    // just set them to some dummy values
    gr.thetas().setRandom();
    gr.radii().setRandom();

    kbs_t kbs(n_arm_samples, gr);

    // v is used in this test,
    // but any values should make this test work
    colvec_type<value_t> v(d);
    v.setRandom();

    mat_type<value_t, 2, 2> A;
    A << 2, 1, 1, 1;
    A *= n_arm_samples;

    value_t expected =
        v.dot(A * v) + v.squaredNorm() * 3 * std::sqrt(n_arm_samples);
    value_t actual = kbs.hessian_quadform_bound(0, 0, v);
    EXPECT_DOUBLE_EQ(actual, expected);
}

}  // namespace exponential
}  // namespace model
}  // namespace imprint
