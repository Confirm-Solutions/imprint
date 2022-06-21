#include <imprint_bits/grid/grid_range.hpp>
#include <imprint_bits/grid/hyperplane.hpp>
#include <imprint_bits/grid/tile.hpp>
#include <imprint_bits/model/exponential/simple_log_rank.hpp>
#include <testutil/base_fixture.hpp>
#include <testutil/model/exponential/simple_log_rank.hpp>
#include <vector>

namespace imprint {
namespace model {
namespace exponential {

struct slr_fixture : base_fixture {
   protected:
    using value_t = double;
    using gen_t = std::mt19937;
    using uint_t = uint32_t;
    using tile_t = grid::Tile<value_t>;
    using gr_t = grid::GridRange<value_t, uint_t, tile_t>;
    using hp_t = grid::HyperPlane<value_t>;

    using model_t = SimpleLogRank<value_t>;
    using model_legacy_t = legacy::ExpControlkTreatment<value_t, uint_t, gr_t>;

    size_t seed = 382;
    gen_t gen;
    value_t cv = 1.96;
    colvec_type<value_t> cvs;
    const size_t n_params = 2;
    const size_t n_gridpts = 2;
    gr_t gr;
    size_t n_arm_samples = 132;
    value_t censor_time = 2;

    slr_fixture() : gen(seed), cvs(1), gr(n_params, n_gridpts) {}

   public:
    void SetUp() override {
        // set threshold
        cvs[0] = cv;

        // initialize grid
        gr.thetas().setRandom();
        auto& radii = gr.radii();
        radii.setRandom();
        radii.array() = (radii.array() + 2) / 2;  // makes it > 0
        std::vector<hp_t> null_hypos;             // no slicing
        gr.create_tiles(null_hypos);
    }
};

TEST_F(slr_fixture, simulate_test) {
    // New model
    model_t model(n_arm_samples, censor_time, cvs);
    auto sgs = model.make_sim_global_state<gen_t, value_t, uint_t>(gr);
    auto ss = sgs.make_sim_state(seed);
    colvec_type<uint_t> actual(gr.n_tiles());
    ss->simulate(actual);

    // Old model
    model_legacy_t model_leg(n_arm_samples, censor_time, cvs);
    model_leg.set_grid_range(gr);
    auto ss_leg = model_leg.make_sim_state(seed);
    colvec_type<uint_t> expected(gr.n_tiles());
    expected.setZero();
    ss_leg->simulate(expected);

    expect_eq_vec(actual, expected);
}

}  // namespace exponential
}  // namespace model
}  // namespace imprint
