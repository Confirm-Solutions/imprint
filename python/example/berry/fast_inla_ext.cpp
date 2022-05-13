#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cmath>
#include <iostream>

namespace py = pybind11;

using Arr = py::array_t<double>;

template <typename T, int D>
struct Access {
    py::buffer_info buf;
    T* ptr;
    py::detail::unchecked_mutable_reference<T, D> get;
};

template <typename T, int D>
Access<T, D> get(Arr x) {
    auto buf = x.request();
    T* ptr = static_cast<T*>(buf.ptr);
    return Access<T, D>{std::move(buf), ptr, x.mutable_unchecked<D>()};
}

int inla_inference(Arr sigma2_post_out, Arr exceedances_out, Arr theta_max_out,
                   Arr theta_sigma_out, Arr y_in, Arr n_in, Arr sigma2_pts_in,
                   Arr sigma2_wts_in, Arr log_prior_in, Arr neg_precQ_in,
                   Arr cov_in, Arr logprecQdet_in, double mu_0, double logit_p1,
                   double tol, Arr thresh_theta_in) {
    auto sigma2_post = get<double, 2>(sigma2_post_out);
    auto exceedances = get<double, 2>(exceedances_out);
    auto theta_max = get<double, 3>(theta_max_out);
    auto theta_sigma = get<double, 3>(theta_sigma_out);

    auto y = get<double, 2>(y_in);
    auto n = get<double, 2>(n_in);
    auto sigma2_pts = get<double, 1>(sigma2_pts_in);
    auto sigma2_wts = get<double, 1>(sigma2_wts_in);
    auto log_prior = get<double, 1>(log_prior_in);
    auto neg_precQ = get<double, 3>(neg_precQ_in);

    auto cov = get<double, 3>(cov_in);
    auto logprecQdet = get<double, 1>(logprecQdet_in);

    int N = y.buf.shape[0];
    int nsig2 = theta_max.buf.shape[1];
    double tol2 = tol * tol;
    auto thresh_theta = get<double, 1>(thresh_theta_in);

#pragma omp parallel for
    for (int p = 0; p < N; p++) {
        for (int s = 0; s < nsig2; s++) {
            std::array<double, 4> t{};
            std::array<double, 4> grad;
            std::array<double, 4> hess_diag;
            std::array<double, 4> step;

            std::array<std::array<double, 4>, 4> tmp;
            std::array<std::array<double, 4>, 4> hess_inv;
            bool converged = false;
            for (int opt_iter = 0; opt_iter < 20; opt_iter++) {
                // construct gradient and hessian.
                for (int i = 0; i < 4; i++) {
                    auto theta_adj = t[i] + logit_p1;
                    auto exp_theta_adj = std::exp(theta_adj);
                    auto C = 1.0 / (exp_theta_adj + 1);
                    auto nCeta = n.get(p, i) * C * exp_theta_adj;
                    grad[i] = y.get(p, i) - nCeta;
                    for (int j = 0; j < 4; j++) {
                        grad[i] += neg_precQ.get(s, i, j) * (t[j] - mu_0);
                    }

                    for (int j = 0; j < 4; j++) {
                        hess_inv[i][j] = cov.get(s, i, j);
                    }
                    hess_diag[i] = nCeta * C;
                }

                // invert hessian by repeatedly using the Sherman-Morrison
                // formula.
                for (int k = 0; k < 4; k++) {
                    double offset =
                        hess_diag[k] / (1 + hess_diag[k] * hess_inv[k][k]);
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            tmp[i][j] =
                                offset * hess_inv[k][i] * hess_inv[j][k];
                        }
                    }
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            hess_inv[i][j] -= tmp[i][j];
                        }
                    }
                }

                // take newton step
                double step_len2 = 0.0;
                for (int i = 0; i < 4; i++) {
                    step[i] = 0.0;
                    for (int j = 0; j < 4; j++) {
                        step[i] += hess_inv[i][j] * grad[j];
                    }
                    step_len2 += step[i] * step[i];
                    t[i] += step[i];
                }

                // check for convergence.
                if (step_len2 < tol2) {
                    converged = true;
                    break;
                }
            }

            // marginal std dev is the sqrt of the hessian diagonal.
            for (int i = 0; i < 4; i++) {
                theta_max.get(p, s, i) = t[i];
                theta_sigma.get(p, s, i) = std::sqrt(hess_inv[i][i]);
            }

            if (!converged) {
                throw std::runtime_error(
                    "INLA optimization failed to converge");
            }

            // calculate log joint distribution
            std::array<double, 4> tmm0;
            for (int i = 0; i < 4; i++) {
                tmm0[i] = t[i] - mu_0;
            }
            double logjoint = logprecQdet.get(s) + log_prior.get(s);
            for (int i = 0; i < 4; i++) {
                double quadsum = 0.0;
                for (int j = 0; j < 4; j++) {
                    quadsum += neg_precQ.get(s, i, j) * tmm0[j];
                }
                logjoint += 0.5 * tmm0[i] * quadsum;

                auto theta_adj = t[i] + logit_p1;
                logjoint += theta_adj * y.get(p, i) -
                            n.get(p, i) * std::log(std::exp(theta_adj) + 1);
            }

            // determinant of hessian (this destroys hess_inv!)
            double c;
            double det = 1;
            for (int i = 0; i < 4; i++) {
                for (int k = i + 1; k < 4; k++) {
                    c = hess_inv[k][i] / hess_inv[i][i];
                    for (int j = i; j < 4; j++) {
                        hess_inv[k][j] = hess_inv[k][j] - c * hess_inv[i][j];
                    }
                }
            }
            for (int i = 0; i < 4; i++) {
                det *= hess_inv[i][i];
            }

            // calculate p(sigma^2 | y)
            sigma2_post.get(p, s) = std::exp(logjoint + 0.5 * std::log(det));
        }

        // normalize the sigma2_post distribution
        double sigma2_integral = 0.0;
        for (int s = 0; s < nsig2; s++) {
            sigma2_integral += sigma2_post.get(p, s) * sigma2_wts.get(s);
        }
        double inv_sigma2_integral = 1.0 / sigma2_integral;
        for (int s = 0; s < nsig2; s++) {
            sigma2_post.get(p, s) *= inv_sigma2_integral;
        }

        // calculate exceedance probabilities, integrated over sigma2
        for (int i = 0; i < 4; i++) {
            exceedances.get(p, i) = 0.0;
            for (int s = 0; s < nsig2; s++) {
                double mu = theta_max.get(p, s, i);
                double normalized =
                    (thresh_theta.get(i) - mu) / theta_sigma.get(p, s, i);
                double exc_sigma2 = 0.5 * (erf(-normalized * M_SQRT1_2) + 1);
                exceedances.get(p, i) +=
                    exc_sigma2 * sigma2_post.get(p, s) * sigma2_wts.get(s);
            }
        }
    }
    return 0;
}

PYBIND11_MODULE(fast_inla_ext, m) { m.def("inla_inference", &inla_inference); }
/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'].extend(['-O3', '-fopenmp'])
cfg['extra_link_args'].extend(['-fopenmp'])
%>
*/
