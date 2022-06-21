#include <Eigen/Core>
#include <imprint_bits/util/algorithm.hpp>
#include <testutil/base_fixture.hpp>

namespace imprint {

struct algorithm_fixture
    : base_fixture,
      testing::WithParamInterface<std::tuple<size_t, size_t, size_t, size_t> > {
   protected:
    Eigen::MatrixXd x;
    Eigen::VectorXd thr;

    algorithm_fixture() {
        size_t seed, n, p, d;
        std::tie(seed, n, p, d) = GetParam();
        srand(seed);
        x.setRandom(n, p);
        thr.setRandom(d);
        sort_cols(thr);
    }
};

TEST_P(algorithm_fixture, sort_cols_test) {
    Eigen::MatrixXd expected = x;
    for (int i = 0; i < x.cols(); ++i) {
        auto expected_i = expected.col(i);
        std::sort(expected_i.data(), expected_i.data() + expected_i.size());
    }
    sort_cols(x);
    expect_double_eq_mat(x, expected);
}

TEST_P(algorithm_fixture, accum_count_test) {
    Eigen::MatrixXi actual(thr.size(), x.cols());
    Eigen::MatrixXi expected(thr.size(), x.cols());
    for (int j = 0; j < expected.cols(); ++j) {
        for (int i = 0; i < expected.rows(); ++i) {
            expected(i, j) = (x.col(j).array() < thr(i)).count();
        }
    }

    sort_cols(x);
    accum_count(x, thr, actual);

    expect_eq_mat(actual, expected);
}

TEST_P(algorithm_fixture, accum_count_map_test) {
    Eigen::MatrixXi actual(thr.size(), x.cols());
    Eigen::MatrixXi expected(thr.size(), x.cols());
    for (int j = 0; j < expected.cols(); ++j) {
        for (int i = 0; i < expected.rows(); ++i) {
            expected(i, j) = (x.col(j).array() < thr(i)).count();
        }
    }

    sort_cols(x);
    Eigen::Map<Eigen::VectorXd> thr_map(thr.data(), thr.size());
    Eigen::Map<Eigen::MatrixXi> actual_map(actual.data(), actual.rows(),
                                           actual.cols());
    accum_count(x, thr_map, actual_map);

    expect_eq_mat(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(
    AlgorithmSuite, algorithm_fixture,

    // combination of inputs: (seed, n, p)
    testing::Combine(testing::Values(10, 23, 145, 241, 412, 23968, 31),
                     testing::Values(1, 5, 10), testing::Values(1, 5, 10),
                     testing::Values(1, 2, 3, 5, 10, 15, 20)));

}  // namespace imprint
