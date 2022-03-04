#include <RcppEigen.h>
#include <Rcpp.h>
#include <kevlar>
#include "unif_sampler.h"

using namespace Rcpp;
using namespace kevlar;

template <class ModelType>
inline List tune_driver(
        ModelType&& model,
        int start_seed,
        int p_batch_size,
        int n_thr,
        int n_sim,
        double alpha,
        double delta,
        double grid_radius,
        const Eigen::Map<Eigen::VectorXd> thr_vec,
        bool do_progress_bar)
{
    size_t start_seed_ = (start_seed < 0) ? time(0) : start_seed;
    size_t p_batch_size_ = (p_batch_size < 0) ? 
        std::numeric_limits<size_t>::infinity() :
        p_batch_size;
    size_t n_thr_ = (n_thr < 0) ? std::thread::hardware_concurrency() : n_thr;

    double opt_lmda = 0.0;
    std::string err_msg;

    try {
        opt_lmda = tune(
            model, n_sim, alpha, delta, grid_radius, thr_vec, 
            start_seed_, p_batch_size_, 
            pb_ostream(Rcpp_cout_get()), do_progress_bar, n_thr_);
    }
    catch (const kevlar_error& e) {
        err_msg = e.what();
    }

    return List::create(Named("lmda")=opt_lmda,
                        Named("err")=err_msg);
}

template <class ModelType>
inline void fit_driver(
        ModelType&& model,
        int start_seed,
        int p_batch_size,
        int n_thr,
        int n_sim,
        double delta,
        double grid_radius,
        double thr,
        String serialize_fname,
        bool do_progress_bar)
{
    size_t start_seed_ = (start_seed < 0) ? time(0) : start_seed;
    size_t p_batch_size_ = (p_batch_size < 0) ? 
        std::numeric_limits<size_t>::infinity() :
        p_batch_size;
    size_t n_thr_ = (n_thr < 0) ? std::thread::hardware_concurrency() : n_thr;
    fit(model, n_sim, delta, grid_radius, thr, 
        serialize_fname.get_cstring(), 
        start_seed_, p_batch_size_, 
        pb_ostream(Rcpp_cout_get()), do_progress_bar, n_thr_);
}

// Tune function for BCKT model.
// [[Rcpp::export]]
List bckt_tune(
        int n_sim,
        double alpha,
        double delta,
        int ph2_size,
        int n_samples,
        int grid_dim,
        double grid_radius,
        const Eigen::Map<Eigen::VectorXd> p,
        const Eigen::Map<Eigen::MatrixXd> p_endpt,
        const Eigen::Map<Eigen::VectorXd> lmda_grid,
        int start_seed=-1,
        int p_batch_size=-1,
        int n_thr=-1,
        bool do_progress_bar=true
        )
{
    // TODO: generalize hypos
    std::vector<std::function<bool(const dAryInt&)> > hypos;
    hypos.reserve(grid_dim-1);
    for (size_t i = 0; i < grid_dim-1; ++i) {
        hypos.emplace_back(
                [i, &p](const dAryInt& mean_idxer) {
                    auto& bits = mean_idxer();
                    return p[bits[i+1]] <= p[bits[0]];
                });
    }

    BinomialControlkTreatment<grid::Rectangular> 
        bckt(grid_dim, ph2_size, n_samples, p, p_endpt, hypos);

    return tune_driver(bckt, start_seed, p_batch_size, n_thr, n_sim, alpha, delta,
                        grid_radius, lmda_grid, do_progress_bar);
}

// Fit function for BCKT model.
// [[Rcpp::export]]
void bckt_fit(
        int n_sim,
        double delta,
        int ph2_size,
        int n_samples,
        int grid_dim,
        double grid_radius,
        const Eigen::Map<Eigen::VectorXd> p,
        const Eigen::Map<Eigen::MatrixXd> p_endpt,
        double lmda,
        String serialize_fname,
        int start_seed=-1,
        int p_batch_size=-1,
        int n_thr=-1,
        bool do_progress_bar=true
        )
{

    // TODO: generalize hypos
    std::vector<std::function<bool(const dAryInt&)> > hypos;
    hypos.reserve(grid_dim-1);
    for (size_t i = 0; i < grid_dim-1; ++i) {
        hypos.emplace_back(
                [i, &p](const dAryInt& mean_idxer) {
                    auto& bits = mean_idxer();
                    return p[bits[i+1]] <= p[bits[0]];
                });
    }

    BinomialControlkTreatment<grid::Rectangular> 
        bckt(grid_dim, ph2_size, n_samples, p, p_endpt, hypos);
    fit_driver(bckt, start_seed, p_batch_size, n_thr, n_sim, delta, grid_radius,
                lmda, serialize_fname, do_progress_bar);
}

// Tune function for ECKT model.
// [[Rcpp::export]]
List eckt_tune(
        int n_sim,
        double alpha,
        double delta,
        int n_samples,
        double grid_radius,
        double censor_time,
        const Eigen::Map<Eigen::VectorXd> lmda,
        const Eigen::Map<Eigen::VectorXd> lmda_lower,
        const Eigen::Map<Eigen::VectorXd> hzrd_rate,
        const Eigen::Map<Eigen::VectorXd> hzrd_rate_lower,
        const Eigen::Map<Eigen::VectorXd> thr_vec,
        int start_seed=-1,
        int p_batch_size=-1,
        int n_thr=-1,
        bool do_progress_bar=true
        )
{
    ExpControlkTreatment<grid::Rectangular> 
        model(n_samples, censor_time, lmda, lmda_lower, hzrd_rate, hzrd_rate_lower);

    return tune_driver(model, start_seed, p_batch_size, n_thr, n_sim, alpha, delta,
                        grid_radius, thr_vec, do_progress_bar);
}

// Fit function for ECKT model.
// [[Rcpp::export]]
void eckt_fit(
        int n_sim,
        double delta,
        int n_samples,
        double grid_radius,
        double censor_time,
        const Eigen::Map<Eigen::VectorXd> lmda,
        const Eigen::Map<Eigen::VectorXd> lmda_lower,
        const Eigen::Map<Eigen::VectorXd> hzrd_rate,
        const Eigen::Map<Eigen::VectorXd> hzrd_rate_lower,
        double thr,
        String serialize_fname,
        int start_seed=-1,
        int p_batch_size=-1,
        int n_thr=-1,
        bool do_progress_bar=true
        )
{
    ExpControlkTreatment<grid::Rectangular> 
        model(n_samples, censor_time, lmda, lmda_lower, hzrd_rate, hzrd_rate_lower);
    fit_driver(model, start_seed, p_batch_size, n_thr, n_sim, delta, 
                grid_radius, thr, serialize_fname, do_progress_bar);
}

// Unserialize output from fitting.
// [[Rcpp::export]]
List unserialize(String fname)
{
    using upper_bd_t = UpperBound<double>;

    UnSerializer us(fname.get_cstring());

    Eigen::VectorXd c;
    Eigen::VectorXd c_bd;
    Eigen::MatrixXd grad;
    Eigen::VectorXd grad_bd;
    Eigen::VectorXd hess_bd;

    upper_bd_t::unserialize(
            us, c, c_bd, grad, grad_bd, hess_bd);

    return List::create(Named("c")=c,
                Named("c_bd")=c_bd,
                Named("grad")=grad,
                Named("grad_bd")=grad_bd,
                Named("hess_bd")=hess_bd);
}
