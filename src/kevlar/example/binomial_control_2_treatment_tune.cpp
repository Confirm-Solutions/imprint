#include <iostream>
#include <kevlar_bits/model/driver.hpp>
#include <kevlar_bits/model/binomial_control_k_treatment.hpp>
#include <kevlar_bits/util/grid.hpp>

int main()
{
    using namespace kevlar;
    using grid_t = Gridder<grid::Rectangular>;

    size_t p_size = 16;
    double lower = -0.5 / 8;
    double upper = 1.5 / 8;
    size_t n_sim = 100000;
    double alpha = 0.025;
    double delta = 0.025;
    size_t grid_dim = 3;
    double grid_radius = grid_t::radius(p_size, lower, upper);
    size_t n_samples = 250;
    size_t ph2_size = 50;

    Eigen::VectorXd p_1d = grid_t::make_grid(p_size, lower, upper);
    p_1d = p_1d.unaryExpr([](auto x) { return 1./(1. + std::exp(-x)); });

    Eigen::MatrixXd p_endpt = grid_t::make_endpts(p_size, lower, upper);
    p_endpt = p_endpt.unaryExpr([](auto x) { return 1./(1. + std::exp(-x)); });

    Eigen::VectorXd thr_vec = grid_t::make_grid(20, 1.95, 5.);
    sort_cols(thr_vec, std::greater<double>());

    std::vector<std::function<bool(const dAryInt&)> > hypos;
    hypos.reserve(grid_dim-1);
    for (size_t i = 0; i < grid_dim-1; ++i) {
        hypos.emplace_back(
                [i, &p_1d](const dAryInt& mean_idxer) { 
                    auto& bits = mean_idxer(); 
                    return p_1d[bits[i+1]] <= p_1d[bits[0]];
                });
    }

    BinomialControlkTreatment<grid::Rectangular> 
        model(grid_dim, ph2_size, n_samples, p_1d, p_endpt, hypos);

    try {
        auto thr = tune(
                model, n_sim, alpha, delta, grid_radius,
                thr_vec, 0, 1000,
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
