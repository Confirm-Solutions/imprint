#include <imprint_bits/distribution/binomial.hpp>
#include <imprint_bits/util/types.hpp>
#include <testutil/base_fixture.hpp>
#include <testutil/eigen_ext.hpp>

namespace imprint {
namespace distribution {

struct binomial_fixture : base_fixture {
   protected:
    using dist_t = Binomial<int>;
};

// ==============================================
// TEST score
// ==============================================

using binomial_score_input_t = std::tuple<int, int, double, double>;

struct binomial_score_fixture
    : binomial_fixture,
      ::testing::WithParamInterface<binomial_score_input_t> {};

TEST_P(binomial_score_fixture, score_test) {
    auto [t, n, p, e] = GetParam();
    auto actual = dist_t::score(t, n, p);
    EXPECT_DOUBLE_EQ(actual, e);

    // test array-like inputs also
    colvec_type<int> tv(3);
    colvec_type<double> pv(3);
    tv.array() = t;
    pv.array() = p;
    colvec_type<double> actual_v =
        dist_t::score(tv.template cast<double>(), n, pv);
    colvec_type<double> expected_v(3);
    expected_v.array() = e;
    expect_double_eq_vec(actual_v, expected_v);
}

INSTANTIATE_TEST_SUITE_P(
    BinomialScoreTest, binomial_score_fixture,
    testing::Values(binomial_score_input_t({3, 10, 0.5, -2}),
                    binomial_score_input_t({3, 5, 0.5, 0.5}),
                    binomial_score_input_t({5, 2, 0.3, 4.4})));

// ==============================================
// TEST Covariance quadratic form
// ==============================================

using binomial_covquad_input_t =
    std::tuple<int, colvec_type<double>, colvec_type<double>, double>;

struct binomial_covquad_fixture
    : binomial_fixture,
      ::testing::WithParamInterface<binomial_covquad_input_t> {};

TEST_P(binomial_covquad_fixture, covar_quadform_test) {
    auto [n, p, v, e] = GetParam();
    auto actual = dist_t::covar_quadform(n, p.array(), v.array());
    EXPECT_DOUBLE_EQ(actual, e);

    // test if n is a vector also
    colvec_type<double> nv(p.size());
    nv.array() = n;
    actual = dist_t::covar_quadform(nv.array(), p.array(), v.array());
    EXPECT_DOUBLE_EQ(actual, e);
}

INSTANTIATE_TEST_SUITE_P(
    BinomialCovarQuadformTest, binomial_covquad_fixture,
    testing::Values(binomial_covquad_input_t(250, make_colvec({0.3, 0.4, 0.5}),
                                             make_colvec({1., 1., 1.}), 175),
                    binomial_covquad_input_t(30, make_colvec({0.1, 0.1}),
                                             make_colvec({0.3, 0.5}), 0.918)));

// ==============================================
// TEST Natural parameter to mean parameter
// ==============================================

using binomial_natural_to_mean_input_t =
    std::tuple<colvec_type<double>, colvec_type<double> >;

struct binomial_natural_to_mean_fixture
    : binomial_fixture,
      ::testing::WithParamInterface<binomial_natural_to_mean_input_t> {};

TEST_P(binomial_natural_to_mean_fixture, covar_quadform_test) {
    auto [t, e] = GetParam();
    colvec_type<double> actual = dist_t::natural_to_mean(t.array());
    expect_double_eq_vec(actual, e);

    // test if n is a scalar also
    auto actual_s = dist_t::natural_to_mean(t[0]);
    EXPECT_DOUBLE_EQ(actual_s, e[0]);
}

INSTANTIATE_TEST_SUITE_P(BinomialNatToMeanTest,
                         binomial_natural_to_mean_fixture,
                         testing::Values(binomial_natural_to_mean_input_t(
                             make_colvec({-0.5, 0., 0.5}),
                             make_colvec({0.3775406687981454353611, 0.5,
                                          0.6224593312018545646389}))));

}  // namespace distribution
}  // namespace imprint
