#include <imprint_bits/bound/accumulator/typeI_error_accum.hpp>
#include <imprint_bits/grid/grid_range.hpp>
#include <imprint_bits/grid/tile.hpp>
#include <testutil/base_fixture.hpp>

namespace imprint {
namespace bound {

struct MockGen {};

template <class GridRangeType>
struct MockSimState {
    MockSimState(size_t n_models, size_t n_gridpts, size_t n_params,
                 const GridRangeType& grid_range)
        : n_models_(n_models),
          n_gridpts_(n_gridpts),
          n_params_(n_params),
          gr_(grid_range) {}

    void simulate(MockGen, colvec_type<uint32_t>& v) {
        for (size_t i = 0; i < n_gridpts_; ++i) {
            v[i] = i % n_models_;
        }
    }

    void score(colvec_type<double>& v, const colvec_type<uint32_t>&) const {
        Eigen::Map<mat_type<double> > vm(v.data(), n_params_, n_gridpts_);
        for (int j = 0; j < vm.rows(); ++j) {
            for (int k = 0; k < vm.cols(); ++k) {
                vm(j, k) = static_cast<double>(j) * k - n_params_;
            }
        }
    }

    void score(size_t j, colvec_type<double>& out) const {
        for (size_t k = 0; k < n_params_; ++k) {
            out[k] = static_cast<double>(k) * j - n_params_;
        }
    }

    auto n_gridpts() const { return n_gridpts_; }
    auto n_tiles(size_t) const { return 1; }
    const auto& grid_range() const { return gr_; }

   private:
    size_t n_models_;
    size_t n_gridpts_;
    size_t n_params_;
    const GridRangeType& gr_;
};

struct MockGridRange {
    MockGridRange(size_t d, size_t n) : d_{d}, n_{n} {}

    auto n_gridpts() const { return n_; }
    auto n_params() const { return d_; }
    bool is_regular(size_t) const { return true; }
    auto n_tiles(size_t) const { return 1; }
    auto n_tiles() const { return n_; }

   private:
    size_t d_;
    size_t n_;
};

bool null_hypo() { return true; }

struct intersum_fixture : base_fixture {
   protected:
    using value_t = double;
    using uint_t = uint32_t;
    using accum_t = TypeIErrorAccum<value_t, uint_t>;
};

TEST_F(intersum_fixture, default_ctor) { accum_t is; }

TEST_F(intersum_fixture, ctor) { accum_t is(0, 0, 0); }

struct test_update_fixture
    : intersum_fixture,
      testing::WithParamInterface<std::tuple<size_t, size_t, size_t> > {
   protected:
    using gr_t = MockGridRange;
    using state_t = MockSimState<gr_t>;
    using gen_t = MockGen;  // dummy generator object
};

TEST_P(test_update_fixture, test_update) {
    size_t n_models;
    size_t n_gridpts;
    size_t n_params;

    std::tie(n_models, n_gridpts, n_params) = GetParam();

    gen_t gen;
    gr_t gr(n_params, n_gridpts);
    state_t mms(n_models, n_gridpts, n_params, gr);
    accum_t accum(n_models, gr.n_tiles(), n_params);
    colvec_type<uint_t> rej_len(gr.n_tiles());
    mms.simulate(gen, rej_len);
    accum.update(rej_len, mms, gr);

    colvec_type<uint_t> v(n_gridpts);
    mms.simulate(gen, v);
    colvec_type<value_t> s(n_params * n_gridpts);
    mms.score(s, v);

    // check Type I sums
    auto& tis = accum.typeI_sum();
    mat_type<uint_t> expected_tis(n_models, n_gridpts);
    for (int j = 0; j < tis.cols(); ++j) {
        for (int i = 0; i < tis.rows(); ++i) {
            expected_tis(i, j) = (static_cast<uint_t>(tis.rows() - i) <= v[j]);
        }
    }
    expect_eq_mat(tis, expected_tis);

    // check score sums
    colvec_type<value_t> expected_score(n_models * n_params * n_gridpts);
    Eigen::Map<mat_type<value_t> > sm(s.data(), n_params, n_gridpts);
    for (size_t j = 0; j < n_gridpts; ++j) {
        Eigen::Map<mat_type<value_t> > expected_score_j(
            expected_score.data() + j * n_models * n_params, n_models,
            n_params);
        for (size_t k = 0; k < n_params; ++k) {
            for (size_t i = 0; i < n_models; ++i) {
                expected_score_j(i, k) = sm(k, j) * (tis.rows() - i <= v[j]);
            }
        }
    }
    auto& score_sum = accum.score_sum();
    expect_eq_vec(score_sum, expected_score);
}

INSTANTIATE_TEST_SUITE_P(
    TestUpdateSuite, test_update_fixture,

    // combination of inputs: (n_models, n_gridpts, n_params)
    testing::Combine(testing::Values(1, 10), testing::Values(1, 5, 15),
                     testing::Values(1, 3, 7, 18)));

}  // namespace bound
}  // namespace imprint
