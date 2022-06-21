#include <imprint_bits/grid/gridder.hpp>
#include <testutil/base_fixture.hpp>

namespace imprint {
namespace grid {

// TEST grid
struct grid_fixture : base_fixture,
                      testing::WithParamInterface<
                          std::tuple<size_t, std::pair<double, double> > > {
    void SetUp() override {
        auto&& sub_param = std::tie(lower, upper);
        std::tie(n, sub_param) = GetParam();
    }

   protected:
    using grid_t = Gridder;
    static constexpr double tol = 2e-15;

    size_t n;
    double lower, upper;
};

TEST_P(grid_fixture, radius_test) {
    auto r = grid_t::radius(n, lower, upper);
    EXPECT_DOUBLE_EQ(upper - lower, 2 * r * n);
}

TEST_P(grid_fixture, make_grid_test) {
    Eigen::VectorXd grid = grid_t::make_grid(n, lower, upper);
    EXPECT_EQ(grid.size(), n);
    auto r = grid[0] - lower;
    for (int i = 1; i < grid.size(); ++i) {
        auto diam = grid[i] - grid[i - 1];
        EXPECT_NEAR(diam, 2. * r, tol);
    }
    EXPECT_NEAR(r, upper - grid[grid.size() - 1], tol);
}

TEST_P(grid_fixture, make_endpts_test) {
    Eigen::MatrixXd endpts = grid_t::make_endpts(n, lower, upper);
    EXPECT_EQ(endpts.rows(), 2);
    EXPECT_EQ(endpts.cols(), n);
    auto r = (endpts(1, 0) - lower) / 2;

    EXPECT_NEAR(endpts(0, 0), lower, tol);
    EXPECT_NEAR(endpts(1, endpts.cols() - 1), upper, tol);
    for (int i = 1; i < endpts.cols(); ++i) {
        for (int k = 0; k < endpts.rows(); ++k) {
            auto diam = endpts(k, i) - endpts(k, i - 1);
            EXPECT_NEAR(diam, 2 * r, tol);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    GridSuite, grid_fixture,
    testing::Combine(testing::Values(1, 2, 3, 5, 10),
                     testing::Values(std::make_pair(-2., 1.),
                                     std::make_pair(-3., 0.),
                                     std::make_pair(1., 1.3),
                                     std::make_pair(0., 0.001),
                                     std::make_pair(-10., -0.3))));

}  // namespace grid
}  // namespace imprint
