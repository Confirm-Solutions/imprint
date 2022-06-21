#include <imprint_bits/grid/grid_range.hpp>
#include <imprint_bits/grid/hyperplane.hpp>
#include <imprint_bits/grid/tile.hpp>
#include <testutil/base_fixture.hpp>
#include <testutil/grid/tile.hpp>

namespace imprint {
namespace grid {

struct grid_range_fixture : base_fixture {
   protected:
    using value_t = double;
    using uint_t = uint32_t;
    using tile_t = Tile<value_t>;
    using hp_t = HyperPlane<value_t>;
    using bits_t = unsigned char;
    using gr_t = GridRange<value_t, uint_t, tile_t>;
    using vec_surf_t = std::vector<hp_t>;

    template <class BitType>
    bool is_null(const BitType& bit, size_t j) {
        return (bit & (1 << j)) == 0;
    }
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
    bits.emplace_back(0 << 0 | 0 << 1);

    // (F, T)
    expected.emplace_back(gr.thetas().col(0), gr.radii().col(0));
    buff << -1, -1;
    expected.back().emplace_back(buff);
    buff << -1, 0;
    expected.back().emplace_back(buff);
    buff << 0, 0;
    expected.back().emplace_back(buff);
    bits.emplace_back(1 << 0 | 0 << 1);

    // (T, F)
    expected.emplace_back(gr.thetas().col(0), gr.radii().col(0));
    buff << -1, -1;
    expected.back().emplace_back(buff);
    buff << 0, -1;
    expected.back().emplace_back(buff);
    buff << 0, 0;
    expected.back().emplace_back(buff);
    bits.emplace_back(0 << 0 | 1 << 1);

    // (F, F)
    expected.emplace_back(gr.thetas().col(0), gr.radii().col(0));
    buff << -1, -1;
    expected.back().emplace_back(buff);
    buff << -1, 0;
    expected.back().emplace_back(buff);
    buff << 0, 0;
    expected.back().emplace_back(buff);
    bits.emplace_back(1 << 0 | 1 << 1);

    // check each of the expected tiles
    for (auto it = tiles.begin(); it != std::next(tiles.begin(), gr.n_tiles(0));
         ++it) {
        EXPECT_NE(std::find(expected.begin(), expected.end(), *it),
                  expected.end());
    }
    for (size_t i = 0; i < bits.size(); ++i) {
        for (size_t j = 0; j < sizeof(bits_t) * 8; ++j) {
            EXPECT_EQ(is_null(bits[i], j), gr.check_null(i, j));
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
    bits.emplace_back(0 << 0 | 0 << 1);

    // (F, T) tile
    expected.emplace_back(gr.thetas().col(3), gr.radii().col(3));
    buff << 0, 0;
    expected.back().emplace_back(buff);
    buff << 0, 1;
    expected.back().emplace_back(buff);
    buff << 1, 1;
    expected.back().emplace_back(buff);
    bits.emplace_back(1 << 0 | 0 << 1);

    // check each of the expected tiles
    auto beg = std::next(tiles.begin(), pos);
    for (auto it = beg; it != std::next(beg, gr.n_tiles(3)); ++it) {
        EXPECT_NE(std::find(expected.begin(), expected.end(), *it),
                  expected.end());
    }
    for (size_t i = 0; i < bits.size(); ++i) {
        for (size_t j = 0; j < (sizeof(bits_t) * 8); ++j) {
            EXPECT_EQ(is_null(bits[i], j), gr.check_null(pos + i, j));
        }
    }
}

TEST_F(grid_range_fixture, prune_points) {
    // COPIED SETTING FROM create_tiles TEST

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
    gr.prune();

    size_t pos = 0;

    const auto& tiles = gr.tiles();
    colvec_type<value_t> buff(d);
    std::vector<tile_t> expected;
    std::vector<bits_t> bits;

    // check tiles for bottom left gridpt
    EXPECT_EQ(gr.n_tiles(0), 3);

    // lower left tile splits:

    // (T, T)
    expected.emplace_back(gr.thetas().col(0), gr.radii().col(0));
    buff << -1, -1;
    expected.back().emplace_back(buff);
    buff << 0, -1;
    expected.back().emplace_back(buff);
    buff << 0, 0;
    expected.back().emplace_back(buff);
    bits.emplace_back(0 << 0 | 0 << 1);

    // (F, T)
    expected.emplace_back(gr.thetas().col(0), gr.radii().col(0));
    buff << -1, -1;
    expected.back().emplace_back(buff);
    buff << -1, 0;
    expected.back().emplace_back(buff);
    buff << 0, 0;
    expected.back().emplace_back(buff);
    bits.emplace_back(1 << 0 | 0 << 1);

    // (T, F)
    expected.emplace_back(gr.thetas().col(0), gr.radii().col(0));
    buff << -1, -1;
    expected.back().emplace_back(buff);
    buff << 0, -1;
    expected.back().emplace_back(buff);
    buff << 0, 0;
    expected.back().emplace_back(buff);
    bits.emplace_back(0 << 0 | 1 << 1);

    // check each of the expected tiles
    for (auto it = tiles.begin(); it != std::next(tiles.begin(), gr.n_tiles(0));
         ++it) {
        EXPECT_NE(std::find(expected.begin(), expected.end(), *it),
                  expected.end());
    }
    for (size_t i = 0; i < bits.size(); ++i) {
        for (size_t j = 0; j < sizeof(bits_t) * 8; ++j) {
            EXPECT_EQ(is_null(bits[i], j), gr.check_null(i, j));
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
    bits.emplace_back(0 << 0 | 0 << 1);

    // (F, T) tile
    expected.emplace_back(gr.thetas().col(3), gr.radii().col(3));
    buff << 0, 0;
    expected.back().emplace_back(buff);
    buff << 0, 1;
    expected.back().emplace_back(buff);
    buff << 1, 1;
    expected.back().emplace_back(buff);
    bits.emplace_back(1 << 0 | 0 << 1);

    // check each of the expected tiles
    auto beg = std::next(tiles.begin(), pos);
    for (auto it = beg; it != std::next(beg, gr.n_tiles(3)); ++it) {
        EXPECT_NE(std::find(expected.begin(), expected.end(), *it),
                  expected.end());
    }
    for (size_t i = 0; i < bits.size(); ++i) {
        for (size_t j = 0; j < (sizeof(bits_t) * 8); ++j) {
            EXPECT_EQ(is_null(bits[i], j), gr.check_null(pos + i, j));
        }
    }
}

TEST_F(grid_range_fixture, prune_off_gridpt) {
    size_t d = 2, n = 1;
    gr_t gr(d, n);
    gr.thetas().col(0) << -0.5, -0.5;
    gr.radii().fill(0.5);

    colvec_type<value_t> normal(d);

    vec_surf_t vs;
    normal << 1, 1;
    normal /= normal.norm();
    vs.emplace_back(normal, 0);

    gr.create_tiles(vs);
    gr.prune();

    EXPECT_EQ(gr.thetas().size(), 0);
    EXPECT_EQ(gr.radii().size(), 0);
    EXPECT_EQ(gr.sim_sizes().size(), 0);
    EXPECT_EQ(gr.n_tiles(), 0);
    EXPECT_EQ(gr.n_gridpts(), 0);
    EXPECT_EQ(gr.n_params(), d);  // should not have changed
}

TEST_F(grid_range_fixture, prune_is_regular) {
    size_t d = 2, n = 1;
    gr_t gr(d, n);
    gr.thetas().col(0) << 0.0, 0.0;
    gr.radii().fill(0.5);

    colvec_type<value_t> normal(d);

    vec_surf_t vs;
    normal << 1, 1;
    normal /= normal.norm();
    vs.emplace_back(normal, 0);

    gr.create_tiles(vs);

    EXPECT_EQ(gr.n_tiles(), 2);
    EXPECT_FALSE(gr.is_regular(0));
    gr.prune();
    EXPECT_EQ(gr.n_tiles(), 1);
    EXPECT_FALSE(gr.is_regular(0));
}

TEST_F(grid_range_fixture, prune_no_surfaces) {
    size_t d = 2, n = 10;
    gr_t gr(d, n);
    gr.thetas().setRandom();
    gr.radii().fill(0.5);

    vec_surf_t vs;
    gr.create_tiles(vs);
    gr.prune();

    EXPECT_EQ(gr.thetas().cols(), n);
    EXPECT_EQ(gr.radii().cols(), n);
    EXPECT_EQ(gr.sim_sizes().size(), n);
    EXPECT_EQ(gr.n_tiles(), n);
    EXPECT_EQ(gr.n_gridpts(), n);
    EXPECT_EQ(gr.n_params(), d);

    for (size_t i = 0; i < gr.n_tiles(); ++i) {
        for (size_t j = 0; j < gr.max_bits(); ++j) {
            EXPECT_TRUE(gr.check_null(i, j));
        }
    }
}

TEST_F(grid_range_fixture, prune_twice_invariance) {
    size_t d = 3, n = 100;

    gr_t gr(d, n);
    gr.thetas().setRandom();
    auto& r = gr.radii();
    r.setRandom();
    r.array() = (r.array() + 1) * 0.5 + 1;
    auto& ss = gr.sim_sizes();
    ss.setRandom();
    ss.array() = ss.array().max(1).min(100);

    colvec_type<value_t> normal(d);
    normal.setZero();
    vec_surf_t vs;
    normal << 1, -1, 0;
    normal /= normal.norm();
    vs.emplace_back(normal, 0);
    normal << 1, 1, 0;
    normal /= normal.norm();
    vs.emplace_back(normal, -1);
    normal << 1, 0, -1;
    normal /= normal.norm();
    vs.emplace_back(normal, 0.5);

    gr.create_tiles(vs);

    // first prune
    gr.prune();

    auto old_thetas = gr.thetas();
    auto old_radii = gr.radii();
    auto old_ss = gr.sim_sizes();
    auto old_tiles = gr.tiles();

    // second prune
    gr.prune();
    auto& new_thetas = gr.thetas();
    auto& new_radii = gr.radii();
    auto& new_ss = gr.sim_sizes();
    auto& new_tiles = gr.tiles();

    expect_double_eq_mat(old_thetas, new_thetas);
    expect_double_eq_mat(old_radii, new_radii);
    expect_eq_vec(old_ss, new_ss);

    // only check for the vertices
    EXPECT_EQ(old_tiles.size(), new_tiles.size());
    for (size_t i = 0; i < old_tiles.size(); ++i) {
        if (old_tiles[i].is_regular()) continue;
        EXPECT_TRUE(check_vertices(new_tiles[i], new_tiles[i]));
    }
}

}  // namespace grid
}  // namespace imprint
