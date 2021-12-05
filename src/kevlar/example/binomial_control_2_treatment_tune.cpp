#include <iostream>
#include <kevlar_bits/model/binomial_control_k_treatment.hpp>
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
    double delta = 0.025;
    size_t grid_dim = 3;
    double grid_radius = grid_t::radius(p_size, lower, upper);
    size_t n_samples = 250;
    size_t ph2_size = 50;

    Eigen::VectorXd p_1d = grid_t::make_grid(p_size, lower, upper);
    p_1d = p_1d.unaryExpr([](auto x) { return 1./(1. + std::exp(-x)); });

    Eigen::MatrixXd p_endpt = grid_t::make_endpts(p_size, lower, upper);
    p_endpt = p_endpt.unaryExpr([](auto x) { return 1./(1. + std::exp(-x)); });

    Eigen::VectorXd thr_vec = grid_t::make_grid(10, 12., 15.2);
    sort_cols(thr_vec, std::greater<double>());

    BinomialControlkTreatment<grid::Rectangular> 
        model(grid_dim, ph2_size, n_samples, p_1d, p_endpt);

    try {
        auto thr = model.tune(
                n_sim, alpha, delta, grid_radius,
                thr_vec, 0, std::numeric_limits<size_t>::infinity(),
                pb_ostream(std::cout), true
                );
        std::cout << thr << std::endl;
    } 
    catch (const kevlar_error& e) {
        std::cerr << e.what() << std::endl;
    } catch (const std::system_error& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
