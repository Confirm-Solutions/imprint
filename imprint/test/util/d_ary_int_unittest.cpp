#include <Eigen/Core>
#include <imprint_bits/util/d_ary_int.hpp>
#include <imprint_bits/util/math.hpp>  // separately unittested
#include <testutil/base_fixture.hpp>

namespace imprint {

struct d_ary_int_fixture : base_fixture,
                           testing::WithParamInterface<std::tuple<int, int> > {
   protected:
    int d, k;

    d_ary_int_fixture() { std::tie(d, k) = GetParam(); }
};

TEST_P(d_ary_int_fixture, d_ary_int_ctor) {
    dAryInt i(d, k);
    auto actual = (i().array() == 0).count();
    auto expected = k;
    EXPECT_EQ(actual, expected);
}

TEST_P(d_ary_int_fixture, d_ary_int_overflow) {
    dAryInt i(d, k);
    for (int j = 0; j < ipow(d, k); ++j) {
        ++i;
    }
    auto actual = (i().array() == 0).count();
    auto expected = k;
    EXPECT_EQ(actual, expected);
}

TEST_P(d_ary_int_fixture, d_ary_int_incr_5) {
    dAryInt i(d, k);
    if (d <= 5) return;
    for (int j = 0; j < d - 5; ++j) {
        ++i;
    }
    auto& actual = i();
    auto expected = actual;
    expected.setZero();
    if (expected.size() < 1) return;
    expected(expected.size() - 1) = d - 5;
    expect_eq_vec(actual, expected);
}

TEST_P(d_ary_int_fixture, d_ary_int_incr_10) {
    dAryInt i(d, k);
    if (d <= 10) return;
    for (int j = 0; j < d + 10; ++j) {
        ++i;
    }
    auto& actual = i();
    auto expected = actual;
    expected.setZero();
    if (expected.size() < 2) return;
    expected(expected.size() - 1) = 10;
    expected(expected.size() - 2) = 1;
    expect_eq_vec(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(MathSuite, d_ary_int_fixture,

                         // combination of inputs: (d, k)
                         testing::Combine(testing::Values(0, 1, 10, 20),
                                          testing::Values(0, 1, 2, 3, 4)));

}  // namespace imprint
