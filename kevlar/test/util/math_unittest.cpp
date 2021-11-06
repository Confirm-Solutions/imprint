#include <kevlar_bits/util/math.hpp>
#include <testutil/base_fixture.hpp>
#include <Eigen/Core>

namespace kevlar {

// TEST ipow
struct ipow_fixture : 
    base_fixture,
    testing::WithParamInterface<
        std::tuple<double, int> >
{
protected:
    double base;
    int exp;

    ipow_fixture()
    {
        std::tie(base, exp) = GetParam();
    }
};

TEST_P(ipow_fixture, ipow_test)
{
    auto actual = ipow(base, exp);
    auto expected = std::pow(base, static_cast<double>(exp));
    EXPECT_DOUBLE_EQ(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(
    MathSuite, ipow_fixture,

    // combination of inputs: (seed, n, p)
    testing::Combine(
        testing::Values(-2., 1., 0., 1., 2.),
        testing::Values(-3, -2, -1, 0, 1, 2, 3)
        )
);

// TEST qnorm
struct qnorm_fixture : 
    base_fixture,
    testing::WithParamInterface<
        std::tuple<double, double> >
{
protected:
    static constexpr double tol = 2e-9;
};

TEST_P(qnorm_fixture, qnorm_test)
{
    double p, expected;
    std::tie(p, expected) = GetParam();
    double actual = qnorm(p);
    EXPECT_NEAR(actual, expected, tol);
}

INSTANTIATE_TEST_SUITE_P(
    MathSuite, qnorm_fixture,
    testing::Values(
        std::make_pair( 0.01, -2.3263478740408407575),
        std::make_pair( 0.1, -1.2815515655446008125 ),
        std::make_pair( 0.3, -0.52440051270804066696),
        std::make_pair( 0.5, 0.                     ),
        std::make_pair( 0.8, 0.84162123357291440673 ),
        std::make_pair( 0.9, 1.2815515655446008125  ),
        std::make_pair( 0.99, 2.3263478740408407575 )
        )
);

} // namespace kevlar
