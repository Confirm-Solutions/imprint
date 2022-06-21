#include <imprint_bits/grid/hyperplane.hpp>
#include <testutil/base_fixture.hpp>

namespace imprint {
namespace grid {

struct hyperplane_fixture : base_fixture {
    void SetUp() override {
        normal.resize(d);
        normal.fill(1);
        shift = 0.23124;
    }

   protected:
    using value_t = double;
    using hp_t = HyperPlane<value_t>;

    size_t d = 6;
    colvec_type<value_t> normal;
    value_t shift;
};

TEST_F(hyperplane_fixture, find_orient) {
    hp_t hp(normal, shift);

    value_t eps = 1e-14;
    colvec_type<value_t> x(d);

    x = shift * normal / normal.squaredNorm();
    auto ori = hp.find_orient(x);
    EXPECT_EQ(ori, orient_type::on);

    x += eps * normal;
    ori = hp.find_orient(x);
    EXPECT_EQ(ori, orient_type::pos);

    x -= 2 * eps * normal;
    ori = hp.find_orient(x);
    EXPECT_EQ(ori, orient_type::neg);
}

TEST_F(hyperplane_fixture, intersect) {
    colvec_type<value_t> normal(3);
    normal << 0, 0, 1;

    hp_t hp(normal, 0.5);

    colvec_type<value_t> v(3);
    v << 1, 0, 0;
    colvec_type<value_t> dir(3);
    dir << 0, 0, 3;
    value_t expected = 0.5 / 3.0;
    value_t actual = hp.intersect(v, dir);
    EXPECT_DOUBLE_EQ(actual, expected);
}

}  // namespace grid
}  // namespace imprint
