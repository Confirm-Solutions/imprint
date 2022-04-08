#include <kevlar_bits/bound/accumulator/typeI_error_accum.hpp>
#include <kevlar_bits/bound/typeI_error_bound.hpp>
#include <kevlar_bits/grid/grid_range.hpp>
#include <kevlar_bits/grid/hyperplane.hpp>
#include <kevlar_bits/grid/tile.hpp>
#include <testutil/base_fixture.hpp>

namespace kevlar {
namespace bound {

struct MockKevlarBoundState {
    using value_t = double;
    using tile_t = Tile<value_t>;

    MockKevlarBoundState(size_t n_nat_params, size_t n_models)
        : n_nat_params_(n_nat_params), n_models_{n_models} {}

    /*
     * The kevlar bound we will assume is:
     *
     * \sup\limits_{v \in R-\theta}
     *      v^\top \widehat{\nabla f} +
     *      \sqrt{||v||^2 \cdots} +
     *      \frac{1}{2} \norm{v}^2
     *
     *  We will test whether the upper bound computation
     *  is truly invariant to directions on a rectangle R.
     *  Since the sup occurs at the corners,
     *  and for the corners, $||v||^2$ is constant,
     *  it should simply be sup of $v^\top \widehat{\nabla f}$.
     *  And the sup is achieved with $|v|^\top |\widehat{\nabla f}|$,
     *  where the absolute value is element-wise.
     */

    value_t covar_quadform(const Eigen::Ref<const colvec_type<value_t>>&,
                           const Eigen::Ref<const colvec_type<value_t>>& v) {
        return v.squaredNorm();
    }

    value_t hessian_quadform_bound(
        const tile_t&, const Eigen::Ref<const colvec_type<value_t>>& v) {
        return v.squaredNorm();
    }

    void apply_eta_jacobian(const Eigen::Ref<const colvec_type<value_t>>&,
                            const Eigen::Ref<const colvec_type<value_t>>& v,
                            Eigen::Ref<colvec_type<value_t>> out) {
        out = v;
    }

    size_t n_models() const { return n_models_; }
    size_t n_natural_params() const { return n_nat_params_; }

   private:
    size_t n_nat_params_;
    size_t n_models_;
};

struct typeI_error_bound_fixture : base_fixture {
    void SetUp() override {
        gr = gr_t(n_params, n_gridpts);

        gr.thetas().setRandom();
        gr.radii().array() = radius;
        gr.sim_sizes().array() = sim_size;

        std::vector<HyperPlane<value_t>> vhp;
        gr.create_tiles(vhp);

        acc_o.reset(n_models, gr.n_tiles(), n_params);
        acc_o.typeI_sum().setRandom();
        auto scale = std::max(acc_o.typeI_sum().maxCoeff() / sim_size, 1uL);
        acc_o.typeI_sum() /= scale;
        acc_o.typeI_sum().array() =
            acc_o.typeI_sum().array().max(0.0).min(sim_size);
        acc_o.score_sum().setRandom();

        ub.create(kbs, acc_o, gr, delta);
    }

   protected:
    using value_t = value_t;
    using uint_t = uint32_t;
    using tile_t = Tile<value_t>;
    using gr_t = GridRange<value_t, uint_t, tile_t>;
    using accum_t = TypeIErrorAccum<value_t, uint_t>;
    using kb_t = TypeIErrorBound<value_t>;

    value_t delta = 0.025;
    value_t radius = 0.25;
    size_t sim_size = 100;
    size_t n_models = 2;
    size_t n_gridpts = 20;
    size_t n_params = 3;
    kb_t ub;
    MockKevlarBoundState kbs;
    accum_t acc_o;
    gr_t gr;

    typeI_error_bound_fixture() : kbs(n_params, n_models) {}
};

TEST_F(typeI_error_bound_fixture, default_ctor) {}

TEST_F(typeI_error_bound_fixture, delta_0) {
    auto actual = ub.delta_0();
    auto expected = acc_o.typeI_sum().template cast<value_t>() / sim_size;
    expect_double_eq_mat(actual, expected);
}

TEST_F(typeI_error_bound_fixture, delta_0_u) {
    auto d0 = ub.delta_0().array();
    const auto& actual = ub.delta_0_u();
    Eigen::MatrixXd expected =
        Eigen::MatrixXd::NullaryExpr(d0.rows(), d0.cols(), [&](auto i, auto j) {
            return ::stats::qbeta(1 - 0.5 * delta, acc_o.typeI_sum()(i, j) + 1,
                                  sim_size - acc_o.typeI_sum()(i, j)) -
                   d0(i, j);
        });
    expect_double_eq_mat(actual, expected);
}

TEST_F(typeI_error_bound_fixture, delta_1) {
    const auto& actual = ub.delta_1();
    Eigen::MatrixXd expected(n_models, n_gridpts);
    for (size_t i = 0; i < n_gridpts; ++i) {
        Eigen::Map<mat_type<value_t>> score_sum_i(
            acc_o.score_sum().data() + n_models * n_params * i, n_models,
            n_params);
        expected.col(i) =
            score_sum_i.array().abs().rowwise().sum() * radius / sim_size;
    }
    expect_double_eq_mat(actual, expected);
}

TEST_F(typeI_error_bound_fixture, delta_1_u) {
    const auto& actual = ub.delta_1_u();
    mat_type<value_t> expected(actual.rows(), actual.cols());
    for (int j = 0; j < expected.cols(); ++j) {
        expected.col(j).array() = std::sqrt(
            kbs.covar_quadform(gr.thetas().col(j), gr.radii().col(j)) /
            sim_size * (2. / delta - 1));
    }
    expect_double_eq_mat(actual, expected);
}

TEST_F(typeI_error_bound_fixture, delta_2_u) {
    const auto& actual = ub.delta_2_u();
    mat_type<value_t> expected(actual.rows(), actual.cols());
    for (int j = 0; j < expected.cols(); ++j) {
        expected.col(j).array() =
            0.5 * kbs.hessian_quadform_bound(gr.tiles()[j], gr.radii().col(j));
    }
    expect_double_eq_mat(actual, expected);
}

}  // namespace bound
}  // namespace kevlar
