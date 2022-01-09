#include <iostream>
#include <kevlar_bits/model/driver.hpp>
#include <kevlar_bits/model/exp_control_k_treatment.hpp>
#include <kevlar_bits/util/grid.hpp>

int main()
{
    using namespace kevlar;
    using grid_t = Gridder<grid::Rectangular>;

    size_t lmda_size = 8;
    double lower = -0.1/4;
    double upper = 1./4;
    double hzrd_lower = -(upper-lower);
    double hzrd_upper = 0.0;
    size_t n_sim = 1000;
    double alpha = 0.025;
    double delta = 0.025;
    double grid_radius = grid_t::radius(lmda_size, lower, upper);
    size_t n_samples = 250;
    double censor_time = 5.0;

    Eigen::VectorXd lmda_1d = grid_t::make_grid(lmda_size, lower, upper);
    lmda_1d.array() = lmda_1d.array().exp();

    Eigen::VectorXd lmda_lower = grid_t::make_endpts(lmda_size, lower, upper).row(0);
    lmda_lower.array() = lmda_lower.array().exp();

    Eigen::VectorXd hzrd_rate = grid_t::make_grid(lmda_size, hzrd_lower, hzrd_upper);
    hzrd_rate.array() = hzrd_rate.array().exp();
    
    Eigen::VectorXd hzrd_rate_lower = grid_t::make_endpts(lmda_size, hzrd_lower, hzrd_upper).row(0);
    hzrd_rate_lower.array() = hzrd_rate_lower.array().exp();

    Eigen::VectorXd thr_vec = grid_t::make_grid(20, 1.96, 2.1);
    sort_cols(thr_vec, std::greater<double>());

    ExpControlkTreatment<grid::Rectangular> 
        model(n_samples, censor_time, lmda_1d, lmda_lower, hzrd_rate, hzrd_rate_lower);

    try {
        auto thr = tune(
                model, n_sim, alpha, delta, grid_radius,
                thr_vec, 0, std::numeric_limits<size_t>::infinity(),
                pb_ostream(std::cout), true, 1);
        std::cout << thr << std::endl;
    } 
    catch (const kevlar_error& e) {
        std::cerr << e.what() << std::endl;
    } catch (const std::system_error& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
