#include <testutil/base_fixture.hpp>
#include <kevlar_bits/intersum.hpp>

namespace kevlar {

struct MockModelState
{
    MockModelState(
        size_t n_models,
        size_t n_gridpts,
        size_t n_params)
        : n_models_(n_models)
        , n_gridpts_(n_gridpts)
        , n_params_(n_params)
    {}

    void get_rej_len(colvec_type<uint32_t>& v) 
    {
        for (size_t i = 0; i < n_gridpts_; ++i) {
            v[i] = i % n_models_;
        }
    }

    void get_grad(colvec_type<double>& v,
                  const colvec_type<uint32_t>&) 
    {
        Eigen::Map<mat_type<double> > vm(v.data(), n_gridpts_, n_params_);
        for (int k = 0; k < vm.cols(); ++k) {
            for (int j = 0; j < vm.rows(); ++j) {
                vm(j,k) = static_cast<double>(k) * j - n_params_;
            }
        }
    }

    double get_grad(uint32_t j, uint32_t k) 
    {
        return static_cast<double>(k) * j - n_params_;
    }

private:
    size_t n_models_;
    size_t n_gridpts_;
    size_t n_params_;
};

bool null_hypo() { return true; }

struct intersum_fixture
    : base_fixture
{
protected:
};

TEST_F(intersum_fixture, default_ctor)
{
    InterSum<double, uint32_t> is;
}

TEST_F(intersum_fixture, ctor)
{
    InterSum<double, uint32_t> is(0, 0, 0);
}

struct test_update_fixture
    : intersum_fixture
    , testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t> >
{
protected:
};

TEST_P(test_update_fixture, test_update)
{  
    size_t n_models;
    size_t n_gridpts;
    size_t n_params;

    std::tie(n_models, n_gridpts, n_params) = GetParam();

    MockModelState mms(n_models, n_gridpts, n_params);
    InterSum<double, uint32_t> is;
    is.reset(n_models, n_gridpts, n_params);
    is.update(mms);

    colvec_type<uint32_t> v(n_gridpts);
    mms.get_rej_len(v);
    colvec_type<double> g(n_gridpts * n_params);
    mms.get_grad(g, v);

    // check accumulation count
    EXPECT_EQ(is.n_accum(), 1);

    // check Type I sums
    auto& tis = is.type_I_sum();
    mat_type<uint32_t> expected_tis(n_models, n_gridpts);
    for (int j = 0; j < tis.cols(); ++j) {
        for (int i = 0; i < tis.rows(); ++i) {
            expected_tis(i,j) = (static_cast<uint32_t>(tis.rows()-i) <= v[j]);
        }
    }    
    expect_eq_mat(tis, expected_tis);

    // check gradient sums
    auto& gr = is.grad_sum();
    colvec_type<double> expected_gr(n_models * n_gridpts * n_params);
    Eigen::Map<mat_type<double> > gm(
            g.data(), n_gridpts, n_params);
    for (size_t k = 0; k < n_params; ++k) {
        Eigen::Map<mat_type<double> > expected_gr_k(
                expected_gr.data() + k * n_models * n_gridpts,
                n_models, n_gridpts);
        for (size_t j = 0; j < n_gridpts; ++j) {
            for (size_t i = 0; i < n_models; ++i) {
                expected_gr_k(i,j) = gm(j, k) * (tis.rows()-i <= v[j]);
            }
        }
    }
    expect_eq_vec(gr, expected_gr);
}

INSTANTIATE_TEST_SUITE_P(
    TestUpdateSuite, test_update_fixture,

    // combination of inputs: (n_models, n_gridpts, n_params)
    testing::Combine(
        testing::Values(1, 10),
        testing::Values(1, 5, 15),
        testing::Values(1, 3, 7, 18)
        )
);

} // namespace kevlar
