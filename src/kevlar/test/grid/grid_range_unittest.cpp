#include <testutil/base_fixture.hpp>
#include <testutil/grid/tile.hpp>
#include <kevlar_bits/grid/tile.hpp>
#include <kevlar_bits/grid/hyperplane.hpp>
#include <kevlar_bits/grid/grid_range.hpp>

namespace kevlar {

struct grid_range_fixture : base_fixture {
   protected:
    using value_t = double;
    using uint_t = uint32_t;
    using tile_t = Tile<value_t>;
    using hp_t = HyperPlane<value_t>;
    using bits_t = unsigned char;
    using gr_t = GridRange<value_t, uint_t, tile_t>;
    using vec_surf_t = std::vector<hp_t>;
};

TEST_F(grid_range_fixture, default_ctor) { gr_t gr; }

TEST_F(grid_range_fixture, ctor) {
    size_t d = 3, n = 10;
    gr_t gr(d, n);

    // make sure internal metadata is stored correctly
    EXPECT_EQ(gr.n_gridpts(), n);
    EXPECT_EQ(gr.n_params(), d);
}

TEST_F(grid_range_fixture, create_tiles) {
    size_t d = 2, n = 4;
    gr_t gr(d, n);
    gr.thetas().col(0) << -0.5, -0.5;
    gr.thetas().col(1) << -0.5, 0.5;
    gr.thetas().col(2) << 0.5, -0.5;
    gr.thetas().col(3) << 0.5, 0.5;
    gr.radii().fill(0.5);

    colvec_type<value_t> normal(d);

    vec_surf_t vs;
    normal << 1, -1;
    normal /= normal.norm();
    vs.emplace_back(normal, 0);
    normal << 1, 1;
    normal /= normal.norm();
    vs.emplace_back(normal, -1);

    gr.create_tiles(vs);

    size_t pos = 0;

    const auto& tiles = gr.tiles();
    colvec_type<value_t> buff(d);
    std::vector<tile_t> expected;
    std::vector<bits_t> bits;

    // check tiles for bottom left gridpt
    EXPECT_EQ(gr.n_tiles(0), 4);

    // lower left tile splits:

    // (T, T)
    expected.emplace_back(gr.thetas().col(0), gr.radii().col(0));
    buff << -1, -1;
    expected.back().emplace_back(buff);
    buff << 0, -1;
    expected.back().emplace_back(buff);
    buff << 0, 0;
    expected.back().emplace_back(buff);
    bits.emplace_back(1 << 1 | 1 << 0);

    // (F, T)
    expected.emplace_back(gr.thetas().col(0), gr.radii().col(0));
    buff << -1, -1;
    expected.back().emplace_back(buff);
    buff << -1, 0;
    expected.back().emplace_back(buff);
    buff << 0, 0;
    expected.back().emplace_back(buff);
    bits.emplace_back(1 << 1);

    // (T, F)
    expected.emplace_back(gr.thetas().col(0), gr.radii().col(0));
    buff << -1, -1;
    expected.back().emplace_back(buff);
    buff << 0, -1;
    expected.back().emplace_back(buff);
    buff << 0, 0;
    expected.back().emplace_back(buff);
    bits.emplace_back(1 << 0);

    // (F, F)
    expected.emplace_back(gr.thetas().col(0), gr.radii().col(0));
    buff << -1, -1;
    expected.back().emplace_back(buff);
    buff << -1, 0;
    expected.back().emplace_back(buff);
    buff << 0, 0;
    expected.back().emplace_back(buff);
    bits.emplace_back(0);

    // check each of the expected tiles
    for (auto it = tiles.begin(); it != std::next(tiles.begin(), gr.n_tiles(0));
         ++it) {
        EXPECT_NE(std::find(expected.begin(), expected.end(), *it),
                  expected.end());
    }
    for (size_t i = 0; i < bits.size(); ++i) {
        for (size_t j = 0; j < sizeof(bits_t); ++j) {
            EXPECT_EQ(bits[i] & (1 << j), gr.check_null(i, j));
        }
    }

    // check tiles for top left gridpt
    pos += gr.n_tiles(0);
    EXPECT_EQ(gr.n_tiles(1), 1);
    EXPECT_TRUE(tiles[pos].is_regular());
    EXPECT_FALSE(gr.check_null(pos, 0));
    EXPECT_TRUE(gr.check_null(pos, 1));

    // check tiles for bottom right gridpt
    pos += gr.n_tiles(1);
    EXPECT_EQ(gr.n_tiles(2), 1);
    EXPECT_TRUE(tiles[pos].is_regular());
    EXPECT_TRUE(gr.check_null(pos, 0));
    EXPECT_TRUE(gr.check_null(pos, 1));

    // check tiles for top right gridpt
    pos += gr.n_tiles(2);
    EXPECT_EQ(gr.n_tiles(3), 2);
    expected.clear();
    bits.clear();

    // (T, T) tile
    expected.emplace_back(gr.thetas().col(3), gr.radii().col(3));
    buff << 0, 0;
    expected.back().emplace_back(buff);
    buff << 1, 0;
    expected.back().emplace_back(buff);
    buff << 1, 1;
    expected.back().emplace_back(buff);
    bits.emplace_back(1 << 0 | 1 << 1);

    // (F, T) tile
    expected.emplace_back(gr.thetas().col(3), gr.radii().col(3));
    buff << 0, 0;
    expected.back().emplace_back(buff);
    buff << 0, 1;
    expected.back().emplace_back(buff);
    buff << 1, 1;
    expected.back().emplace_back(buff);
    bits.emplace_back(1 << 1);

    // check each of the expected tiles
    auto beg = std::next(tiles.begin(), pos);
    for (auto it = beg; it != std::next(beg, gr.n_tiles(3)); ++it) {
        EXPECT_NE(std::find(expected.begin(), expected.end(), *it),
                  expected.end());
    }
    for (size_t i = 0; i < bits.size(); ++i) {
        for (size_t j = 0; j < sizeof(bits_t); ++j) {
            EXPECT_EQ(bits[i] & (1 << j), gr.check_null(pos + i, j));
        }
    }
}

}  // namespace kevlar
