#include <testutil/base_fixture.hpp>
#include <kevlar_bits/grid/grid_range.hpp>

namespace kevlar {

struct grid_range_fixture:
    base_fixture
{
protected:
};

TEST_F(grid_range_fixture, default_ctor)
{
    GridRange gr;

    EXPECT_EQ(gr.begin(), gr.end());
}

TEST_F(grid_range_fixture, ctor)
{
    size_t d = 3, n = 10;
    GridRange gr(d, n);

    // make sure internal metadata is stored correctly
    EXPECT_EQ(gr.size(), n);
    EXPECT_EQ(gr.dim(), d);

    // make sure iterating ends at the right place
    auto it = gr.begin();
    for (size_t i = 0; i < n; ++i, ++it);
    EXPECT_EQ(it, gr.end());
}

TEST_F(grid_range_fixture, iteration)
{
    size_t d = 3, n = 2;
    GridRange gr(d, n);

    // set underlying values to random 
    gr.get_thetas().setRandom();
    gr.get_radii().setRandom();
    gr.get_sim_sizes().setRandom();
    gr.get_sim_sizes_rem().setRandom();

    auto it = gr.begin();
    for (size_t i = 0; i < n; ++i, ++it) {
        auto true_theta_i = gr.get_thetas().col(i);
        auto true_radius_i = gr.get_radii().col(i);
        expect_eq_vec(it->get_theta(), true_theta_i);
        expect_eq_vec(it->get_radius(), true_radius_i);
        EXPECT_EQ((*it).get_sim_size(), gr.get_sim_sizes()[i]);
        EXPECT_EQ((*it).get_sim_size_rem(), gr.get_sim_sizes_rem()[i]);

        // Just to check if operator!= works
        EXPECT_NE(it, gr.end());
    }
    EXPECT_EQ(it, gr.end());
}

} // namespace kevlar
