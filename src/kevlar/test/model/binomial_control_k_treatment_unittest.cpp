#include <testutil/base_fixture.hpp>
#include <testutil/model/binomial_control_k_treatment_legacy.hpp>
#include <kevlar_bits/model/binomial_control_k_treatment.hpp>
#include <kevlar_bits/util/range/grid_range.hpp>
#include <random>

namespace kevlar {

struct bckt_fixture
    : base_fixture
{
    void SetUp() override 
    {
        // legacy setup
        theta_1d.setRandom(n_thetas);
        std::sort(theta_1d.data(), theta_1d.data()+theta_1d.size());

        prob_1d.array() = sigmoid(theta_1d.array());
        prob_endpt_1d.resize(2, theta_1d.size());
        prob_endpt_1d.row(0).array() = sigmoid(theta_1d.array()-radius);
        prob_endpt_1d.row(1).array() = sigmoid(theta_1d.array()+radius);

        // new setup
        // only thetas and radii need to be populated.
        
        // populate theta as the cartesian product of theta_1d
        auto& thetas = grid_range.get_thetas();
        dAryInt bits(n_thetas, n_arms);
        for (size_t j = 0; j < grid_range.size(); ++j) {
            for (size_t i = 0; i < n_arms; ++i) {
                thetas(i,j) = theta_1d[bits()[i]];
            }
            ++bits;
        }

        // populate radii as fixed radius
        grid_range.get_radii().array() = radius;
    }

protected:
    using value_t = double;
    using int_t = uint32_t;
    using bckt_legacy = legacy::BinomialControlkTreatment;
    using bckt = BinomialControlkTreatment<value_t, int_t>;

    // common configuration
    
    // configuration that may want to be parametrizations
    size_t n_arms = 3;
    size_t ph2_size = 50;
    size_t n_samples = 250;
    value_t threshold = 1.96;
    value_t radius = 0.25;
    size_t n_thetas = 10;

    // configuration for legacy
    colvec_type<value_t> theta_1d;
    colvec_type<value_t> prob_1d;
    mat_type<value_t> prob_endpt_1d;

    // configuration for new
    GridRange<value_t, int_t> grid_range;

    bckt_fixture()
        : theta_1d()
        , prob_1d()
        , prob_endpt_1d()
        , grid_range(n_arms, ipow(n_thetas, n_arms))
    {}
};

TEST_F(bckt_fixture, ctor)
{
    bckt b_new(n_arms, ph2_size, n_samples, grid_range, threshold);
    bckt_legacy b_leg(n_arms, ph2_size, n_samples, prob_1d, prob_endpt_1d);
}

TEST_F(bckt_fixture, tr_cov_test)
{
    dAryInt bits(n_thetas, n_arms);
    bckt b_new(n_arms, ph2_size, n_samples, grid_range, threshold);
    bckt_legacy b_leg(n_arms, ph2_size, n_samples, prob_1d, prob_endpt_1d);
    for (size_t i = 0; i < ipow(n_thetas, n_arms); ++i, ++bits) {
        EXPECT_DOUBLE_EQ(b_new.tr_cov(i), b_leg.tr_cov(bits));
    }
}

TEST_F(bckt_fixture, tr_max_cov_test)
{
    dAryInt bits(n_thetas, n_arms);
    bckt b_new(n_arms, ph2_size, n_samples, grid_range, threshold);
    bckt_legacy b_leg(n_arms, ph2_size, n_samples, prob_1d, prob_endpt_1d);
    for (size_t i = 0; i < ipow(n_thetas, n_arms); ++i, ++bits) {
        EXPECT_DOUBLE_EQ(b_new.tr_max_cov(i), b_leg.tr_max_cov(bits));
    }
}

struct bckt_state_fixture
    : bckt_fixture
{
protected:
    using state_t = bckt::StateType;
    using state_leg_t = bckt_legacy::StateType;

    size_t seed = 3214;
    std::mt19937 gen;

    template <class StateType>
    void state_gen(StateType& s) 
    {
        gen.seed(seed);
        s.gen_rng(gen);
        s.gen_suff_stat();
    }
};

TEST_F(bckt_state_fixture, test_rej)
{
    bckt b_new(n_arms, ph2_size, n_samples, grid_range, threshold);
    bckt_legacy b_leg(n_arms, ph2_size, n_samples, prob_1d, prob_endpt_1d);
    state_t s_new(b_new);
    state_leg_t s_leg(b_leg);

    state_gen(s_new);
    state_gen(s_leg);

    // get legacy rejections
    colvec_type<int_t> expected(ipow(n_thetas, n_arms));
    dAryInt bits(n_thetas, n_arms);
    for (int i = 0; i < expected.size(); ++i, ++bits) {
        expected[i] = (s_leg.test_stat(bits) > threshold);
    }

    // get new rejections
    colvec_type<int_t> actual(expected.size());
    s_new.get_rej_len(actual);

    expect_eq_vec(actual, expected);
}

TEST_F(bckt_state_fixture, grad_test)
{
    bckt b_new(n_arms, ph2_size, n_samples, grid_range, threshold);
    bckt_legacy b_leg(n_arms, ph2_size, n_samples, prob_1d, prob_endpt_1d);
    state_t s_new(b_new);
    state_leg_t s_leg(b_leg);

    state_gen(s_new);
    state_gen(s_leg);

    // get gradient estimates from new
    colvec_type<value_t> actual(grid_range.size() * n_arms);
    s_new.get_grad(actual);

    // get gradient estimates from legacy
    colvec_type<value_t> expected(grid_range.size() * n_arms);
    Eigen::Map<mat_type<value_t> > expected_m(expected.data(), grid_range.size(), n_arms);
    for (size_t j = 0; j < n_arms; ++j) {
        dAryInt bits(n_thetas, n_arms);
        for (size_t i = 0; i < grid_range.size(); ++i, ++bits) {
            expected_m(i,j) = s_leg.grad_lr(j, bits);
        }
    }

    expect_eq_vec(actual, expected);
}

} // namespace kevlar
