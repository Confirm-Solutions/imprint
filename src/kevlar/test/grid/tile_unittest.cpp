#include <testutil/base_fixture.hpp>
#include <kevlar_bits/grid/tile.hpp>
#include <unordered_set>

namespace kevlar {

struct tile_fixture:
    base_fixture
{
    void SetUp() override 
    {
        d = 3;
        center.setRandom(d);
        radius.setRandom(d);
        radius.array() = 0.5*(radius.array()+1) + 0.0001;
    }

protected:
    using value_t = double;
    using tile_t = Tile<value_t>;

    size_t d;
    colvec_type<value_t> center;
    colvec_type<value_t> radius;
};

TEST_F(tile_fixture, ctor)
{
    tile_t tile(center, radius);
}

TEST_F(tile_fixture, is_regular)
{
    tile_t tile(center, radius);

    EXPECT_TRUE(tile.is_regular());

    tile.emplace_back(center); // add dummy
    EXPECT_FALSE(tile.is_regular());

    tile.make_regular();
    EXPECT_TRUE(tile.is_regular());
}

TEST_F(tile_fixture, set_check_null)
{
    tile_t tile(center, radius);

    std::unordered_set<size_t> idx = {
        0, 1, 4, 7
    };

    for (auto i : idx) tile.set_null(i, true);
    for (size_t i = 0; i < tile_t::n_bits; ++i) {
        if (idx.find(i) != idx.end()) {
            EXPECT_TRUE(tile.check_null(i));
        } else {
            EXPECT_FALSE(tile.check_null(i));
        }
    }

    // unset the rest of the bits to false
    // result should be the same
    for (size_t i = 0; i < tile_t::n_bits; ++i) {
        if (idx.find(i) == idx.end()) {
            tile.set_null(i, false);
        }
    }
    for (size_t i = 0; i < tile_t::n_bits; ++i) {
        if (idx.find(i) != idx.end()) {
            EXPECT_TRUE(tile.check_null(i));
        } else {
            EXPECT_FALSE(tile.check_null(i));
        }
    }

    // unset the idxes
    for (auto i : idx) tile.set_null(i, false);
    for (size_t i = 0; i < tile_t::n_bits; ++i) {
        EXPECT_FALSE(tile.check_null(i));
    }
}

TEST_F(tile_fixture, full_iter)
{
    tile_t tile(center, radius);

    dAryInt bits(2, d);
    colvec_type<value_t> expected(d);

    for (auto it = tile.begin_full();
            it != tile.end_full();
            ++it, ++bits)
    {
        expected = center + 
            ((2*bits().template cast<value_t>().array()-1)
                * radius.array()).matrix();
        auto& v = *it;
        expect_double_eq_vec(v, expected);
    }
}

} // namespace kevlar
