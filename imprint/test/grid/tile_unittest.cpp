#include <imprint_bits/grid/tile.hpp>
#include <testutil/base_fixture.hpp>
#include <unordered_set>

namespace imprint {
namespace grid {

struct tile_fixture : base_fixture {
    void SetUp() override {
        d = 3;
        center.setRandom(d);
        radius.setRandom(d);
        radius.array() = 0.5 * (radius.array() + 1) + 0.0001;
    }

   protected:
    using value_t = double;
    using tile_t = Tile<value_t>;

    size_t d;
    colvec_type<value_t> center;
    colvec_type<value_t> radius;
};

TEST_F(tile_fixture, ctor) { tile_t tile(center, radius); }

TEST_F(tile_fixture, is_regular) {
    tile_t tile(center, radius);

    EXPECT_TRUE(tile.is_regular());

    tile.emplace_back(center);  // add dummy
    EXPECT_FALSE(tile.is_regular());

    tile.make_regular();
    EXPECT_TRUE(tile.is_regular());
}

TEST_F(tile_fixture, full_iter) {
    tile_t tile(center, radius);

    dAryInt bits(2, d);
    colvec_type<value_t> expected(d);

    for (auto it = tile.begin_full(); it != tile.end_full(); ++it, ++bits) {
        expected = center + ((2 * bits().template cast<value_t>().array() - 1) *
                             radius.array())
                                .matrix();
        auto& v = *it;
        expect_double_eq_vec(v, expected);
    }
}

}  // namespace grid
}  // namespace imprint
