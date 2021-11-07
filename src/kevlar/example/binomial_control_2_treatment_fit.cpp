#include <kevlar_bits/model/binomial_control_k_treatment.hpp>
#include <kevlar_bits/process/fit.hpp>
#include <kevlar_bits/util/grid.hpp>

int main()
{
    using namespace kevlar;
    using grid_t = Gridder<grid::Rectangular>;

    size_t p_size = 64;
    double lower = -0.5;
    double upper = 1.5;
    size_t n_sim = 50000;
    double alpha = 0.05;
    double delta = 0.05;
    size_t grid_dim = 3;
    size_t grid_radius = grid_t::radius(p_size, lower, upper);
    size_t n_samples = 250;
    size_t ph2_size = 50;

    Eigen::VectorXd p_1d = grid_t::make_grid(p_size, lower, upper);
    p_1d = p_1d.unaryExpr([](auto x) { return 1./(1. + std::exp(-x)); });

    Eigen::MatrixXd p_endpt = grid_t::make_endpts(p_size, lower, upper);
    p_endpt = p_endpt.unaryExpr([](auto x) { return 1./(1. + std::exp(-x)); });

    auto rng_gen_f = [=](auto& gen, auto& rng) {
        std::uniform_real_distribution<double> unif(0., 1.);
        rng = Eigen::MatrixXd::NullaryExpr(n_samples, grid_dim, 
                [&](auto, auto) { return unif(gen); });
    };

    BinomialControlkTreatment<grid::Rectangular> 
        model(grid_dim, ph2_size, n_samples);

    try {
        fit(n_sim, alpha, delta, grid_dim, grid_radius,
             p_1d, p_endpt, 13.552, rng_gen_f, model, "fit_out", 0);
    } 
    catch (const kevlar_error& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
