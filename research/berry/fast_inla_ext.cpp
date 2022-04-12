#include <array>
#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

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
                   Arr y_in, Arr n_in, Arr sigma2_pts_in, Arr sigma2_wts_in,
                   Arr log_prior_in, Arr neg_precQ_in, Arr cov_in,
                   Arr precQdet_in, double mu_0, double logit_p1) {
    auto sigma2_post = get<double, 2>(sigma2_post_out);
    auto exceedances = get<double, 2>(exceedances_out);
    auto theta_max = get<double, 3>(theta_max_out);

    auto y = get<double, 2>(y_in);
    auto n = get<double, 2>(n_in);
    auto sigma2_pts = get<double, 1>(sigma2_pts_in);
    auto sigma2_wts = get<double, 1>(sigma2_wts_in);
    auto log_prior = get<double, 1>(log_prior_in);
    auto neg_precQ = get<double, 3>(neg_precQ_in);

    //TODO: remove?
    auto cov = get<double, 3>(cov_in);
    auto precQdet = get<double, 1>(precQdet_in);
    int N = y.buf.shape[0];
    int nsig2 = theta_max.shape[1];
    int na = 4;

    for (int p = 0; p < N; p++) {
        for (int k = 0; k < nsig2; k++) {
            std::array<double, 4> t{};
            std::array<double, 4> grad;
            for (int i = 0; i < 4; i++) {
                auto theta_adj = t[i] + logit_p1;
                auto exp_theta_adj = std::exp(theta_adj);
                auto C = 1.0 / (exp_theta_adj + 1);
                grad[i] = y.get(p, i) - (n.get(p, i) * exp_theta_adj) * C;
                for (int j = 0; j < 4; j++) {
                    grad[i] += neg_precQ.get(k, i, j) * (t[j] - mu_0);
                }
            }
        }
        theta_max.ptr[i] = 0.0;
    }
    for (int i = 0; i < N * nsig2 * na; i++) {
    }

    for (int i = 0; i < 100; i++) {
        
    }
    return 0;
}

PYBIND11_MODULE(fast_inla_ext, m) { m.def("inla_inference", &inla_inference); }
/*
<%
setup_pybind11(cfg)
%>
*/