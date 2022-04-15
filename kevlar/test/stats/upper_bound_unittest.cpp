#include <kevlar_bits/grid/grid_range.hpp>
#include <kevlar_bits/grid/hyperplane.hpp>
#include <kevlar_bits/grid/tile.hpp>
#include <kevlar_bits/stats/inter_sum.hpp>
#include <kevlar_bits/stats/upper_bound.hpp>
#include <testutil/base_fixture.hpp>

namespace kevlar {

struct MockModel {
    MockModel(size_t n_models) : n_models_{n_models} {}

    double cov_quad(size_t j,
                    const Eigen::Ref<const colvec_type<double>>&) const {
        return j;
    }

    double max_cov_quad(size_t j,
                        const Eigen::Ref<const colvec_type<double>>&) const {
        return j;
    }

    void eta_transform(size_t, const Eigen::Ref<const colvec_type<double>>& x,
                       colvec_type<double>& out) const {
        out = x;
    }
    double max_eta_hess_cov(size_t) const { return 0; }

    size_t n_models() const { return n_models_; }

   private:
    size_t n_models_;
};

struct upper_bound_fixture : base_fixture {
    void SetUp() override {
        gr = GridRange<double, uint32_t, Tile<double>>(n_params, n_gridpts);

        gr.thetas().setRandom();
        gr.radii().array() = radius;
        gr.sim_sizes().array() = sim_size;

        std::vector<HyperPlane<double>> vhp;
        gr.create_tiles(vhp);

        is_o = InterSum<double, uint32_t>(n_models, gr.n_tiles(), n_params);
        is_o.type_I_sum().setRandom();
        auto scale = std::max(is_o.type_I_sum().maxCoeff() / sim_size, 1uL);
        is_o.type_I_sum() /= scale;
        is_o.type_I_sum().array() =
            is_o.type_I_sum().array().max(0.0).min(sim_size);
        is_o.grad_sum().setRandom();

        ub.create(model, is_o, gr, delta);
    }

   protected:
    double delta = 0.025;
    double radius = 0.25;
    size_t sim_size = 100;
    size_t n_models = 2;
    size_t n_gridpts = 20;
    size_t n_params = 3;
    UpperBound<double> ub;
    MockModel model;
    InterSum<double, uint32_t> is_o;
    GridRange<double, uint32_t, Tile<double>> gr;

    upper_bound_fixture() : model(n_models) {}
};

TEST_F(upper_bound_fixture, default_ctor) {}

TEST_F(upper_bound_fixture, delta_0) {
    auto actual = ub.delta_0();
    auto expected = is_o.type_I_sum().template cast<double>() / sim_size;
    expect_double_eq_mat(actual, expected);
}

TEST_F(upper_bound_fixture, delta_0_u) {
    auto d0 = ub.delta_0().array();
    const auto& actual = ub.delta_0_u();
    Eigen::MatrixXd expected =
        Eigen::MatrixXd::NullaryExpr(d0.rows(), d0.cols(), [&](auto i, auto j) {
            return ::stats::qbeta(1 - 0.5 * delta, is_o.type_I_sum()(i, j) + 1,
                                  sim_size - is_o.type_I_sum()(i, j)) -
                   d0(i, j);
        });
    expect_double_eq_mat(actual, expected);
}

TEST_F(upper_bound_fixture, delta_1) {
    const auto& actual = ub.delta_1();
    Eigen::Map<mat_type<double>> grad_sum(is_o.grad_sum().data(),
                                          n_models * n_gridpts, n_params);
    Eigen::VectorXd grad_l1 =
        grad_sum.array().abs().rowwise().sum() * radius / sim_size;
    Eigen::Map<Eigen::MatrixXd> expected(grad_l1.data(), n_models, n_gridpts);
    expect_double_eq_mat(actual, expected);
}

TEST_F(upper_bound_fixture, delta_1_u) {
    const auto& actual = ub.delta_1_u();
    mat_type<double> expected(actual.rows(), actual.cols());
    for (int j = 0; j < expected.cols(); ++j) {
        expected.col(j).array() = std::sqrt(
            model.cov_quad(j, gr.radii().col(j)) / sim_size * (2. / delta - 1));
    }
    expect_double_eq_mat(actual, expected);
}

TEST_F(upper_bound_fixture, delta_2_u) {
    const auto& actual = ub.delta_2_u();
    mat_type<double> expected(actual.rows(), actual.cols());
    for (int j = 0; j < expected.cols(); ++j) {
        expected.col(j).array() =
            0.5 * model.max_cov_quad(j, gr.radii().col(j));
    }
    expect_double_eq_mat(actual, expected);
}

}  // namespace kevlar
