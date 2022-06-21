#include <benchmark/benchmark.h>

#include <imprint_bits/bound/accumulator/typeI_error_accum.hpp>
#include <imprint_bits/bound/typeI_error_bound.hpp>
#include <imprint_bits/grid/grid_range.hpp>
#include <imprint_bits/grid/hyperplane.hpp>
#include <imprint_bits/grid/tile.hpp>
#include <imprint_bits/model/binomial/common/fixed_n_default.hpp>
#include <vector>

namespace imprint {
namespace {

static void BM_imprint_bound(benchmark::State& state) {
    using namespace model::binomial;
    using value_t = double;
    using uint_t = uint32_t;
    using tile_t = grid::Tile<value_t>;
    using gr_t = grid::GridRange<value_t, uint_t, tile_t>;
    using hp_t = grid::HyperPlane<value_t>;
    using kbs_t = ImprintBoundStateFixedNDefault<gr_t>;
    using kb_t = bound::TypeIErrorBound<value_t>;
    using acc_t = bound::TypeIErrorAccum<value_t, uint_t>;

    size_t n_models = 1;
    size_t n_tiles = 180000;
    size_t n_params = 3;
    size_t n_arm_samples = 21;  // arbitrary
    value_t alpha = 0.025;
    value_t delta = 0.025;

    std::vector<hp_t> hps;

    gr_t gr(n_params, n_tiles);
    gr.create_tiles(hps);

    acc_t acc_o(n_models, n_tiles, n_params);

    auto& typeI_sum = acc_o.typeI_sum__();
    typeI_sum.setRandom();
    typeI_sum /= typeI_sum.maxCoeff() / alpha;
    auto& score_sum = acc_o.score_sum__();
    score_sum.setRandom();

    kbs_t kbs(n_arm_samples, gr);
    kb_t kb;

    for (auto _ : state) {
        kb.create(kbs, acc_o, gr, delta);
    }
}

BENCHMARK(BM_imprint_bound);

}  // namespace
}  // namespace imprint
