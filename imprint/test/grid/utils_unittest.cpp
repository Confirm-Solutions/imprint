#include <imprint_bits/grid/hyperplane.hpp>
#include <imprint_bits/grid/tile.hpp>
#include <imprint_bits/grid/utils.hpp>
#include <testutil/base_fixture.hpp>

namespace imprint {
namespace grid {

struct utils_fixture : base_fixture {
   protected:
    using value_t = double;
    using tile_t = Tile<value_t>;
    using hp_t = HyperPlane<value_t>;

    colvec_type<value_t> center;
    colvec_type<value_t> radius;
    colvec_type<value_t> normal;
    value_t shift;

    template <class RT>
    void check_is_oriented(RT run_test) {
        size_t d = 2;
        center.setZero(d);
        radius.resize(d);
        radius.fill(1);
        normal.setOnes(d);
        shift = 0;

        // should cut through along y = -x line
        run_test(center, radius, normal, shift, false, orient_type::none);

        // check hyperplane shifted to top right corner
        // first, shift slightly below
        value_t eps = 1e-14;
        value_t n_sq = normal.squaredNorm();
        shift += (1 - eps) * n_sq;
        run_test(center, radius, normal, shift, false, orient_type::none);

        // next, shift to exactly the top right corner
        // now entering negative orientation
        shift += eps * n_sq;
        run_test(center, radius, normal, shift, true, orient_type::non_pos);

        // next, go slightly beyond top right corner
        // should end up still in negative orientation
        shift += 2 * n_sq;
        run_test(center, radius, normal, shift, true, orient_type::non_pos);

        // check positive side
        shift = -(1 - eps) * n_sq;
        run_test(center, radius, normal, shift, false, orient_type::none);

        shift -= eps * n_sq;
        run_test(center, radius, normal, shift, true, orient_type::non_neg);

        shift -= 2 * n_sq;
        run_test(center, radius, normal, shift, true, orient_type::non_neg);
    }

    template <class Tile, class Vertices>
    void is_vertices_same(const Tile& t, const Vertices& expected) {
        size_t count = 0;
        for (auto& x : t) {
            auto it = std::find_if(
                expected.begin(), expected.end(),
                [&](const auto& v) { return (v.array() == x.array()).all(); });
            EXPECT_NE(it, expected.end());
            ++count;
        }
        EXPECT_EQ(count, expected.size());
    }
};

TEST_F(utils_fixture, test_is_oriented_shift_inv) {
    // just go through a lot of examples
    for (int i = 0; i < 100; ++i) {
        size_t d = 6;
        center.setRandom(d);
        radius.setRandom(d);
        radius.array() = 0.5 * (radius.array() + 1) + 1e-8;
        normal.setRandom(d);
        shift = 0.3897874;

        // perturbation direction
        colvec_type<value_t> pert_dir(d);
        pert_dir.setRandom();

        // get initial output
        bool expected;
        orient_type expected_ori;
        {
            tile_t tile(center, radius);
            hp_t hp(normal, shift);
            expected = is_oriented(tile, hp, expected_ori);
        }

        // get output after perturbation
        bool actual;
        orient_type actual_ori;
        {
            center += pert_dir;
            shift += normal.dot(pert_dir);
            tile_t tile(center, radius);
            hp_t hp(normal, shift);
            actual = is_oriented(tile, hp, actual_ori);
        }

        EXPECT_EQ(actual, expected);
        EXPECT_EQ(actual_ori, expected_ori);
    }
}

TEST_F(utils_fixture, test_is_oriented_full) {
    auto run_test = [](const auto& center, const auto& radius,
                       const auto& normal, auto shift, bool expected,
                       orient_type exp_ori) {
        tile_t tile(center, radius);
        hp_t hp(normal, shift);
        orient_type ori;
        bool actual = is_oriented(tile, hp, ori);
        if (expected)
            EXPECT_TRUE(actual);
        else
            EXPECT_FALSE(actual);
        EXPECT_EQ(ori, exp_ori);
    };

    check_is_oriented(run_test);
}

TEST_F(utils_fixture, test_is_oriented) {
    auto run_test = [](const auto& center, const auto& radius,
                       const auto& normal, auto shift, bool expected,
                       orient_type exp_ori) {
        tile_t tile(center, radius);
        for (auto it = tile.begin_full(); it != tile.end_full(); ++it) {
            tile.emplace_back(*it);
        }
        hp_t hp(normal, shift);
        orient_type ori;
        bool actual = is_oriented(tile, hp, ori);
        if (expected)
            EXPECT_TRUE(actual);
        else
            EXPECT_FALSE(actual);
        EXPECT_EQ(ori, exp_ori);
    };

    check_is_oriented(run_test);
}

TEST_F(utils_fixture, test_intersect_d2) {
    size_t d = 2;
    center.setZero(d);
    radius.resize(d);
    radius.fill(1);
    normal.setOnes(d);
    shift = 0;

    colvec_type<value_t> buff(d);
    std::vector<colvec_type<value_t>> n_expected;
    std::vector<colvec_type<value_t>> p_expected;

    tile_t p_tile(center, radius);
    tile_t n_tile(center, radius);

    auto run_test = [&]() {
        tile_t tile(center, radius);
        hp_t hp(normal, shift);
        intersect(tile, hp, p_tile, n_tile);
        is_vertices_same(n_tile, n_expected);
        is_vertices_same(p_tile, p_expected);
    };

    // test when shift = 0

    // non-positive region
    buff << -1, -1;
    n_expected.push_back(buff);
    buff << -1, 1;
    n_expected.push_back(buff);
    buff << 1, -1;
    n_expected.push_back(buff);

    // non-negative region
    buff << -1, 1;
    p_expected.push_back(buff);
    buff << 1, -1;
    p_expected.push_back(buff);
    buff << 1, 1;
    p_expected.push_back(buff);

    run_test();

    // test slightly more non-trivial shift
    shift = 0.75 * normal.squaredNorm();
    n_expected.clear();
    p_expected.clear();

    // non-positive region
    buff << -1, -1;
    n_expected.push_back(buff);
    buff << -1, 1;
    n_expected.push_back(buff);
    buff << 1, -1;
    n_expected.push_back(buff);
    buff << 0.5, 1;
    n_expected.push_back(buff);
    buff << 1, 0.5;
    n_expected.push_back(buff);

    // non-negative region
    buff << 0.5, 1;
    p_expected.push_back(buff);
    buff << 1, 0.5;
    p_expected.push_back(buff);
    buff << 1, 1;
    p_expected.push_back(buff);

    run_test();
}

TEST_F(utils_fixture, test_intersect_d3) {
    size_t d = 3;
    center.setZero(d);
    radius.resize(d);
    radius.fill(1);
    normal.setOnes(d);
    shift = 0;

    colvec_type<value_t> buff(d);
    std::vector<colvec_type<value_t>> n_expected;
    std::vector<colvec_type<value_t>> p_expected;

    tile_t p_tile(center, radius);
    tile_t n_tile(center, radius);

    auto run_test = [&]() {
        tile_t tile(center, radius);
        hp_t hp(normal, shift);
        intersect(tile, hp, p_tile, n_tile);
        is_vertices_same(n_tile, n_expected);
        is_vertices_same(p_tile, p_expected);
    };

    // test when shift = 0

    // non-positive region
    buff << 1, -1, -1;
    n_expected.push_back(buff);
    buff << -1, 1, -1;
    n_expected.push_back(buff);
    buff << -1, -1, 1;
    n_expected.push_back(buff);
    buff << -1, -1, -1;
    n_expected.push_back(buff);

    // non-negative region
    buff << -1, 1, 1;
    p_expected.push_back(buff);
    buff << 1, -1, 1;
    p_expected.push_back(buff);
    buff << 1, 1, -1;
    p_expected.push_back(buff);
    buff << 1, 1, 1;
    p_expected.push_back(buff);

    // intersections
    buff << 1, 0, -1;
    n_expected.push_back(buff);
    p_expected.push_back(buff);
    buff << 1, -1, 0;
    n_expected.push_back(buff);
    p_expected.push_back(buff);
    buff << 0, 1, -1;
    n_expected.push_back(buff);
    p_expected.push_back(buff);
    buff << -1, 1, 0;
    n_expected.push_back(buff);
    p_expected.push_back(buff);
    buff << 0, -1, 1;
    n_expected.push_back(buff);
    p_expected.push_back(buff);
    buff << -1, 0, 1;
    n_expected.push_back(buff);
    p_expected.push_back(buff);

    run_test();
}

TEST_F(utils_fixture, test_intersect_d2_non_reg) {
    size_t d = 2;
    center.setZero(d);
    radius.resize(d);
    radius.fill(1);
    normal.setOnes(d);
    shift = 0;

    colvec_type<value_t> buff(d);
    std::vector<colvec_type<value_t>> n_expected;
    std::vector<colvec_type<value_t>> p_expected;

    tile_t p_tile(center, radius);
    tile_t n_tile(center, radius);

    auto run_test = [&]() {
        tile_t tile(center, radius);
        for (auto it = tile.begin_full(); it != tile.end_full(); ++it) {
            tile.emplace_back(*it);
        }
        hp_t hp(normal, shift);
        intersect(tile, hp, p_tile, n_tile);
        is_vertices_same(n_tile, n_expected);
        is_vertices_same(p_tile, p_expected);
    };

    // test when shift = 0
    for (auto it = p_tile.begin_full(); it != p_tile.end_full(); ++it) {
        n_expected.push_back(*it);
    }
    p_expected = n_expected;

    run_test();

    // test slightly more non-trivial shift
    shift = 0.5 * normal.squaredNorm();

    run_test();
}

// EXAMPLES THAT FAILED IN APPLICATION

TEST_F(utils_fixture, test_is_oriented_full_issue1) {
    size_t d = 2;
    center.resize(d);
    center << -0.5, -0.5;
    radius.resize(d);
    radius << 0.5, 0.5;
    normal.resize(d);
    normal << 1, -1;
    normal /= normal.norm();
    shift = 0;

    tile_t tile(center, radius);
    hp_t hp(normal, shift);

    orient_type ori;
    bool actual = is_oriented(tile, hp, ori);
    EXPECT_FALSE(actual);
    EXPECT_EQ(ori, orient_type::none);
}

}  // namespace grid
}  // namespace imprint
