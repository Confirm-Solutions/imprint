#include <testutil/base_fixture.hpp>
#include <kevlar_bits/grid/hyperplane.hpp>

namespace kevlar {

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
    hp_t hp(normal, shift);

    colvec_type<value_t> v(d);
    colvec_type<value_t> dir(d);
    v.setRandom();
    dir.setRandom();
    value_t expected = (normal.dot(v + dir) - shift) / (normal.dot(dir));

    value_t actual = hp.intersect(v, dir);
    EXPECT_DOUBLE_EQ(actual, expected);
}

}  // namespace kevlar
