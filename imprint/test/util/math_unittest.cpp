#include <Eigen/Core>
#include <imprint_bits/util/macros.hpp>
#include <imprint_bits/util/math.hpp>
#include <testutil/base_fixture.hpp>

namespace imprint {

// TEST ipow
struct ipow_fixture : base_fixture,
                      testing::WithParamInterface<std::tuple<double, int>> {
   protected:
    double base;
    int exp;

    ipow_fixture() { std::tie(base, exp) = GetParam(); }
};

TEST_P(ipow_fixture, ipow_test) {
    auto actual = ipow(base, exp);
    auto expected = std::pow(base, static_cast<double>(exp));
    EXPECT_DOUBLE_EQ(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(MathSuite, ipow_fixture,

                         // combination of inputs: (seed, n, p)
                         testing::Combine(testing::Values(-2., 1., 0., 1., 2.),
                                          testing::Values(-3, -2, -1, 0, 1, 2,
                                                          3)));

TEST(MathSuite, normal_cdf) {
    Eigen::Vector<double, 11> x = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5};

    // scipy.stats.norm.cdf(np.arange(-5, 6, 1))
    Eigen::Vector<double, 11> want = {
        2.86651572e-07, 3.16712418e-05, 1.34989803e-03, 2.27501319e-02,
        1.58655254e-01, 5.00000000e-01, 8.41344746e-01, 9.77249868e-01,
        9.98650102e-01, 9.99968329e-01, 9.99999713e-01};
    auto got = normal_cdf(x);
    for (int i = 0; i < 11; ++i) {
        EXPECT_NEAR(got(i), want(i), 1e-8);
    }
};
// TEST qnorm
struct qnorm_fixture : base_fixture,
                       testing::WithParamInterface<std::tuple<double, double>> {
   protected:
    static constexpr double tol = 2e-9;
};

TEST_P(qnorm_fixture, qnorm_test) {
    double p, expected;
    std::tie(p, expected) = GetParam();
    double actual = qnorm(p);
    EXPECT_NEAR(actual, expected, tol);
}

INSTANTIATE_TEST_SUITE_P(
    MathSuite, qnorm_fixture,
    testing::Values(std::make_pair(0.01, -2.3263478740408407575),
                    std::make_pair(0.1, -1.2815515655446008125),
                    std::make_pair(0.3, -0.52440051270804066696),
                    std::make_pair(0.5, 0.),
                    std::make_pair(0.8, 0.84162123357291440673),
                    std::make_pair(0.9, 1.2815515655446008125),
                    std::make_pair(0.99, 2.3263478740408407575)));

TEST(MathSuite, invgamma_pdf) {
    const double alpha_prior = 0.0005;
    const double beta_prior = 0.000005;
    Eigen::Vector<double, 50> x = {
        1.0144596452884776e-08, 1.0785099797992792e-08, 1.2037950072042833e-08,
        1.4100869129550908e-08, 1.7323663774844695e-08, 2.2304299894494958e-08,
        3.00654659174565e-08,   4.2381479416476284e-08, 6.239305748305179e-08,
        9.578524408624434e-08,  1.530886238340002e-07,  2.5426192914231663e-07,
        4.379872035345178e-07,  7.808532796832305e-07,  1.4375862176725688e-06,
        2.726649565774876e-06,  5.314717722088802e-06,  1.0618548564125005e-05,
        2.16880888901994e-05,   4.515945064572312e-05,  9.55905442322199e-05,
        0.00020509741997061919, 0.00044473545658897185, 0.0009717155249005231,
        0.0021328442340584966,  0.004688574927467345,   0.010291077731853369,
        0.022485277150371356,   0.04875731738328307,    0.10461285768712429,
        0.22143759184428968,    0.461082580887007,      0.9417482944689066,
        1.8815674741178463,     3.6675046641566267,     6.956104529292073,
        12.806503167991673,     22.83171727233327,      39.32952146525543,
        65.32164016866048,      104.40021420205447,     160.27424209362246,
        235.95212195712938,     332.6075181224376,      448.3440434043004,
        577.2450983792908,      709.1761442593054,      830.7062199256145,
        927.2051429566812,      985.7464559032629};

    Eigen::Vector<double, 50> want = {
        4.386151925929204e-210,  2.1260113143039648e-197,
        1.7150951887795844e-176, 3.5931981000969017e-150,
        1.3016829277868028e-121, 9.89031055009593e-94,
        9.938775676043121e-69,   6.864402811272085e-48,
        1.2641582357022606e-31,  1.1179876666096545e-19,
        2.1404023212123916e-11,  5.677585568252689e-06,
        0.01259848073322265,     1.0617742488988657,
        10.74566054518207,       29.323046023785686,
        36.73031154858327,       29.40143359271028,
        18.299218185268018,      9.903396205257428,
        4.958190464108896,       2.375423489081816,
        1.1095230246358416,      0.5107132797985644,
        0.23323998830413245,     0.10619528098799831,
        0.04839122837587901,     0.02214491275706143,
        0.010209779947518491,    0.004756955900595907,
        0.002246523464732018,    0.0010785230384441366,
        0.0005278622751423179,   0.00026411102450984666,
        0.0001354538297633819,   7.13932489367365e-05,
        3.87668304416398e-05,    2.173836109706042e-05,
        1.2616202875274078e-05,  7.5941662100922005e-06,
        4.750441558540105e-06,   3.0937026089960848e-06,
        2.1010405401837134e-06,  1.4902245259283024e-06,
        1.10536962472565e-06,    8.584278581268307e-07,
        6.986589754076235e-07,   5.963998866518745e-07,
        5.343001778154085e-07,   5.025538816079104e-07};
    auto got = invgamma_pdf(x, alpha_prior, beta_prior);
    EXPECT_TRUE(want.isApprox(got));
};

}  // namespace imprint
