#include <testutil/base_fixture.hpp>
#include <kevlar_bits/model/binomial_control_k_treatment.hpp>
#include <kevlar_bits/util/d_ary_int.hpp>   // separately unittested
#include <kevlar_bits/util/math.hpp>   // separately unittested
#include <kevlar_bits/util/range/grid_range.hpp>   // separately unittested

namespace kevlar {

template <class GridType>
struct bckt_upper_bound_fixture
    : base_fixture
    , testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t> >
{
protected:
    using bckt_t = BinomialControlkTreatment<GridType>;

    Eigen::VectorXd p;
    Eigen::MatrixXi suff_stat;
    Eigen::VectorXd thr_vec;
    size_t ph2_size, k, unif_rows;

    const double alpha = 0.05;
    const double width = 1.96;
    const double grid_radius = 0.01;

    bckt_upper_bound_fixture()
    {
        size_t p_size;
        std::tie(p_size, unif_rows, k) = GetParam();
        p.setRandom(p_size);        
        p = (p.array() + 1) / 2;
        suff_stat.setRandom(p.size(), k);
        ph2_size = unif_rows/4;
        thr_vec.setRandom(5); 
        thr_vec.array() += 1.5;
        sort_cols(thr_vec, std::greater<double>());
    }

    // mock out the test statistic function
    template <class PType>
    static auto test_stat(
            const dAryInt& p_idxer,
            const PType& p) 
    {
        auto& bits = p_idxer();
        return p[bits[bits.size()-1]];
    }
};

using bckt_upper_bound_fixture_rect = 
    bckt_upper_bound_fixture<grid::Rectangular>;

TEST_P(bckt_upper_bound_fixture_rect, update_one_test)
{
    bckt_t opt(k, ph2_size, unif_rows, p, p);

    UpperBound<double> upper_bd;
    upper_bd.reset(thr_vec.size(), 1, k);

    rectangular_range p_range(p.size(), k, 1); 
    upper_bd.update(opt, p_range, thr_vec);
    auto& upper_bd_raw = upper_bd.get();
    auto actual = upper_bd_raw.col(0);
    
    Eigen::VectorXd expected(thr_vec.size());
    auto z = test_stat(p_range.get_idxer(), p);
    expected.array() = (z > thr_vec.array()).template cast<double>();

    expect_double_eq_vec(actual, expected);
}

TEST_P(bckt_upper_bound_fixture_rect, update_two_test)
{
    bckt_t opt(k, ph2_size, unif_rows, );

    auto upper_bd = opt.make_upper_bd();
    upper_bd.reset(thr_vec.size(), 1);

    rectangular_range p_range(p, k, 2); 
    auto it = p_range.begin();
    upper_bd.update(p_range, suff_stat, thr_vec, 
            [&](const dAryInt& p_idxer) { return test_stat(p_idxer, p); });
    ++it;
    p_range.set_idxer(*it);
    upper_bd.update(p_range, suff_stat, thr_vec, 
            [&](const dAryInt& p_idxer) { return test_stat(p_idxer, p); });
    auto& upper_bd_raw = upper_bd.get();
    auto actual = upper_bd_raw.col(0);
    
    rectangular_range p_range_2(p, k, 2);
    it = p_range_2.begin();
    Eigen::VectorXd expected(thr_vec.size());
    auto z = test_stat(*it, p);
    expected.array() = (z > thr_vec.array()).template cast<double>();
    ++it;
    z = test_stat(*it, p);
    expected.array() += (z > thr_vec.array()).template cast<double>();

    expect_double_eq_vec(actual, expected);
}

TEST_P(bckt_upper_bound_fixture_rect, create_two_test)
{
    bckt_t opt(k, ph2_size, unif_rows);

    // do the actual routine
    auto upper_bd = opt.make_upper_bd();
    upper_bd.reset(thr_vec.size(), 1);
    dAryInt it(p.size(), k); 
    Eigen::MatrixXd p_endpt(2, p.size());
    p_endpt.setRandom();
    sort_cols(p_endpt);

    for (int i = 0; i < 2; ++i, ++it) {
        upper_bd.update(
                rectangular_range(p, it, 1), suff_stat, thr_vec, 
                [&](const dAryInt& p_idxer) { return test_stat(p_idxer, p); });
    }
    it.setZero();
    upper_bd.create(rectangular_range(p, it, 1), p_endpt, alpha, width, grid_radius);
    auto& upper_bd_raw = upper_bd.get();
    auto actual = upper_bd_raw.col(0);

    // do the expected routine
    it.setZero();
    Eigen::VectorXd expected(thr_vec.size());
    Eigen::MatrixXd grad_expected(thr_vec.size(), k);
    expected.setZero();
    grad_expected.setZero();

    // update
    for (int a = 0; a < 2; ++a, ++it) {
        auto z = test_stat(it, p);
        expected.array() += (z > thr_vec.array()).template cast<double>() * 0.5;
        for (int i = 0; i < grad_expected.cols(); ++i) {
            auto col = grad_expected.col(i);
            auto grad = suff_stat(it()[i], i) - unif_rows * p[it()[i]];
            col.array() += 0.5 * grad * (z > thr_vec.array()).template cast<double>();
        }
    }

    // add constant upper bound
    expected.array() += width / std::sqrt(2) * (expected.array() * (1-expected.array())).sqrt();

    // add grad term
    for (int a = 0; a < grad_expected.cols(); ++a) {
        expected.array() += grid_radius * grad_expected.col(a).array().abs();
    }

    // add grad upper bound
    expected.array() += 
        grid_radius * std::sqrt(k * unif_rows * p[0] * (1-p[0]) / 2. * (1/alpha - 1));

    // add hessian upper bound
    bool is_upper_max = std::abs(p_endpt(0,0)-0.5) > std::abs(p_endpt(1,0)-0.5);
    auto max_p = p_endpt(is_upper_max, 0);
    expected.array() += grid_radius * grid_radius * unif_rows / 2. * k * max_p * (1-max_p);

    expect_double_eq_vec(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(
    BCKTUpperBoundSuite, bckt_upper_bound_fixture_rect,
    testing::Combine(
        testing::Values(1, 2, 3, 4),
        testing::Values(50, 100, 250, 300),
        testing::Values(2, 3, 4)
        )
);

} // namespace kevlar
