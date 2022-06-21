#include <imprint_bits/distribution/exponential.hpp>
#include <imprint_bits/util/types.hpp>
#include <testutil/base_fixture.hpp>
#include <testutil/eigen_ext.hpp>

namespace imprint {
namespace distribution {

struct exponential_fixture : base_fixture {
   protected:
    using dist_t = Exponential<double>;
};

// ==============================================
// TEST score
// ==============================================

using exponential_score_input_t = std::tuple<double, int, double, double>;

struct exponential_score_fixture
    : exponential_fixture,
      ::testing::WithParamInterface<exponential_score_input_t> {};

TEST_P(exponential_score_fixture, score_test) {
    auto [t, n, l, e] = GetParam();
    auto actual = dist_t::score(t, n, l);
    EXPECT_DOUBLE_EQ(actual, e);

    // test array-like inputs also
    colvec_type<int> tv(3);
    colvec_type<double> lv(3);
    tv.array() = t;
    lv.array() = l;
    colvec_type<double> actual_v =
        dist_t::score(tv.template cast<double>().array(), n, lv.array());
    colvec_type<double> expected_v(3);
    expected_v.array() = e;
    expect_double_eq_vec(actual_v, expected_v);
}

INSTANTIATE_TEST_SUITE_P(
    ExponentialScoreTest, exponential_score_fixture,
    testing::Values(exponential_score_input_t({3., 10, 0.3,
                                               -30.333333333333336}),
                    exponential_score_input_t({3., 5, 0.2, -22.}),
                    exponential_score_input_t({5., 2, 5., 4.6})));

// ==============================================
// TEST Covariance quadratic form
// ==============================================

using exponential_covquad_input_t =
    std::tuple<int, colvec_type<double>, colvec_type<double>, double>;

struct exponential_covquad_fixture
    : exponential_fixture,
      ::testing::WithParamInterface<exponential_covquad_input_t> {};

TEST_P(exponential_covquad_fixture, covar_quadform_test) {
    auto [n, l, v, e] = GetParam();
    auto actual = dist_t::covar_quadform(n, l.array(), v.array());
    EXPECT_DOUBLE_EQ(actual, e);

    // test if n is a vector also
    colvec_type<double> nv(l.size());
    nv.array() = n;
    actual = dist_t::covar_quadform(nv.array(), l.array(), v.array());
    EXPECT_DOUBLE_EQ(actual, e);
}

INSTANTIATE_TEST_SUITE_P(
    ExponentialCovarQuadformTest, exponential_covquad_fixture,
    testing::Values(exponential_covquad_input_t(250,
                                                make_colvec({0.3, 0.4, 0.5}),
                                                make_colvec({1., 1., 1.}),
                                                5340.2777777777774),
                    exponential_covquad_input_t(30, make_colvec({0.1, 0.1}),
                                                make_colvec({0.3, 0.5}),
                                                1020)));

// ==============================================
// TEST Natural parameter to mean parameter
// ==============================================

using exponential_natural_to_mean_input_t =
    std::tuple<colvec_type<double>, colvec_type<double> >;

struct exponential_natural_to_mean_fixture
    : exponential_fixture,
      ::testing::WithParamInterface<exponential_natural_to_mean_input_t> {};

TEST_P(exponential_natural_to_mean_fixture, covar_quadform_test) {
    auto [t, e] = GetParam();
    colvec_type<double> actual = dist_t::natural_to_mean(t.array());
    expect_double_eq_vec(actual, e);

    // test if n is a scalar also
    auto actual_s = dist_t::natural_to_mean(t[0]);
    EXPECT_DOUBLE_EQ(actual_s, e[0]);
}

INSTANTIATE_TEST_SUITE_P(ExponentialNatToMeanTest,
                         exponential_natural_to_mean_fixture,
                         testing::Values(exponential_natural_to_mean_input_t(
                             make_colvec({1., 2., 3.}),
                             make_colvec({-1., -2., -3.}))));

}  // namespace distribution
}  // namespace imprint
