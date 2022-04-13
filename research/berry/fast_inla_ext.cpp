#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

std::array<std::array<double,4>,4> cholesky(std::array<std::array<double,4>,4> mat) {
    std::array<std::array<double,4>,4> lower;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < i; j++) {
            double sum = 0;
            // Evaluating L(i, j) using L(j, j)
            for (int k = 0; k < j; k++) {
                sum += lower[i][k] * lower[j][k];
            }
            lower[i][j] = (mat[i][j] - sum) / lower[j][j];
        }
        double sum = 0;
        for (int k = 0; k < i; k++) {
            sum += lower[i][k] * lower[i][k];
        }
        lower[i][i] = sqrt(mat[i][i] - sum);
    }
    return lower;
}

std::array<double,4> cho_solve(std::array<std::array<double,4>,4> lower, std::array<double,4> b)  {
    std::array<double,4> y;
    for (int i = 0; i < 4; i++) {
        double sum = b[i];
        for (int j = 0; j < i; j++) {
            sum -= lower[i][j] * y[j];
        }
        y[i] = sum / lower[i][i];
    }

    std::array<double,4> x;
    for (int i = 3; i >= 0; i--) {
        double sum = y[i];
        for (int j = 3; j > i; j--) {
            sum -= lower[j][i] * x[j];
        }
        x[i] = sum / lower[i][i];
    }

    return x;
}

int inla_inference(Arr sigma2_post_out, Arr exceedances_out, Arr theta_max_out,
                   Arr y_in, Arr n_in, Arr sigma2_pts_in, Arr sigma2_wts_in,
                   Arr log_prior_in, Arr neg_precQ_in, Arr cov_in,
                   Arr logprecQdet_in, double mu_0, double logit_p1, double tol,
                   double thresh_theta) {
    auto sigma2_post = get<double, 2>(sigma2_post_out);
    auto exceedances = get<double, 2>(exceedances_out);
    auto theta_max = get<double, 3>(theta_max_out);

    auto y = get<double, 2>(y_in);
    auto n = get<double, 2>(n_in);
    auto sigma2_pts = get<double, 1>(sigma2_pts_in);
    auto sigma2_wts = get<double, 1>(sigma2_wts_in);
    auto log_prior = get<double, 1>(log_prior_in);
    auto neg_precQ = get<double, 3>(neg_precQ_in);

    // TODO: remove?
    auto cov = get<double, 3>(cov_in);
    auto logprecQdet = get<double, 1>(logprecQdet_in);
    int N = y.buf.shape[0];
    int nsig2 = theta_max.buf.shape[1];
    int na = 4;
    double tol2 = tol * tol;

    #pragma omp parallel for
    for (int p = 0; p < N; p++) {
        std::vector<double> theta_sigma(nsig2 * 4);
        for (int s = 0; s < nsig2; s++) {
            std::array<double, 4> t{};
            std::array<double, 4> grad;

            std::array<std::array<double, 4>, 4> hess;
            std::array<std::array<double, 4>, 4> hess_cho;
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
                        hess[i][j] = -neg_precQ.get(s, i, j);
                    }
                    hess[i][i] += nCeta * C;
                }


                // solve for newton step
                hess_cho = cholesky(hess);
                auto step = cho_solve(hess_cho, grad);

                // check for convergence.
                double step_len2 = 0.0;
                for (int i = 0; i < 4; i++) {
                    step_len2 += step[i] * step[i];
                    t[i] += step[i];
                }
                if (step_len2 < tol2) {
                    converged = true;
                    break;
                }
            }

            // marginal std dev is the sqrt of the hessian diagonal.
            for (int i = 0; i < 4; i++) {
                std::array<double,4> e{};
                e[i] = 1.0;
                auto hess_inv_row = cho_solve(hess_cho, e);
                theta_sigma[s * 4 + i] = std::sqrt(hess_inv_row[i]);
            }

            assert(converged);
            // calculate log joint distribution
            double logjoint = logprecQdet.get(s) + log_prior.get(s);
            for (int i = 0; i < 4; i++) {
                theta_max.get(p, s, i) = t[i];

                for (int j = 0; j < 4; j++) {
                    logjoint += 0.5 * (t[i] - mu_0) * neg_precQ.get(s, i, j) *
                                (t[j] - mu_0);
                }

                auto theta_adj = t[i] + logit_p1;
                auto exp_theta_adj = std::exp(theta_adj);
                logjoint += theta_adj * y.get(p, i) -
                            n.get(p, i) * std::log(exp_theta_adj + 1);
            }

            // determinant of hessian (this destroys hess_inv!)
            double det = 1;
            for (int i = 0; i < 4; i++) {
                det *= hess_cho[i][i] * hess_cho[i][i];
            }

            // calculate p(sigma^2 | y)
            sigma2_post.get(p, s) = std::exp(logjoint - 0.5 * std::log(det));
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
                    (thresh_theta - mu) / theta_sigma[s * 4 + i];
                double cdf = 0.5 * erfc(-normalized * M_SQRT1_2);
                double exc_sigma2 = 1.0 - cdf;
                exceedances.get(p, i) +=
                    exc_sigma2 * sigma2_post.get(p, s) * sigma2_wts.get(s);
            }
        }
    }
    return 0;
}

PYBIND11_MODULE(fast_inla_ext, m) { 
    m.def("inla_inference", &inla_inference); 
    m.def("cholesky", &cholesky); 
    m.def("cho_solve", &cho_solve); 
}
/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'].extend(['-Ofast', '-fopenmp'])
cfg['extra_link_args'].extend(['-fopenmp'])
%>
*/