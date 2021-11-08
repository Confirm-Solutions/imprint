#include <kevlar>
#include "unif_sampler.h"
#include <RcppEigen.h>

using namespace Rcpp;
using namespace kevlar;

// Tune function for BCKT model.
// [[Rcpp::export]]
double bckt_tune(
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
    size_t start_seed_ = (start_seed < 0) ? time(0) : start_seed;
    size_t p_batch_size_ = (p_batch_size < 0) ? 
        std::numeric_limits<size_t>::infinity() :
        p_batch_size;
    size_t n_thr_ = (n_thr < 0) ? std::thread::hardware_concurrency() : n_thr;

    BinomialControlkTreatment<grid::Rectangular> 
        bckt(grid_dim, ph2_size, n_samples);

    return tune(
        n_sim, alpha, delta, grid_dim, grid_radius, p, p_endpt,
        lmda_grid, [&](auto& gen, auto& rng) { unif_sampler(gen, rng, n_samples, grid_dim); }, bckt, 
        start_seed_, p_batch_size_, pb_ostream(Rcpp_cout_get()), do_progress_bar, n_thr_);
}

// Fit function for BCKT model.
// [[Rcpp::export]]
void bckt_fit(
        int n_sim,
        double alpha,
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
    size_t start_seed_ = (start_seed < 0) ? time(0) : start_seed;
    size_t p_batch_size_ = (p_batch_size < 0) ? 
        std::numeric_limits<size_t>::infinity() :
        p_batch_size;
    size_t n_thr_ = (n_thr < 0) ? std::thread::hardware_concurrency() : n_thr;

    BinomialControlkTreatment<grid::Rectangular> 
        bckt(grid_dim, ph2_size, n_samples);

    fit(n_sim, alpha, delta, grid_dim, grid_radius, p, p_endpt,
        lmda, [&](auto& gen, auto& rng) { unif_sampler(gen, rng, n_samples, grid_dim); }, bckt, 
        serialize_fname.get_cstring(), 
        start_seed_, p_batch_size_, pb_ostream(Rcpp_cout_get()), do_progress_bar, n_thr_);
}

// Unserialize output from fitting.
// [[Rcpp::export]]
List bckt_unserialize(
        String fname
        )
{
    using bckt_t = BinomialControlkTreatment<grid::Rectangular>;
    using upper_bd_t = typename bckt_t::upper_bd_t;

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
