#include <iostream>

#include <gtest/gtest.h>
#include <kevlar_bits/model/direct_bayes_binomial_control_k_treatment.hpp>

namespace kevlar {
namespace {

const double mu_sig_sq = 0.1;
const double alpha_prior = 0.0005;
const double beta_prior = 0.000005;
const int n_points = 50;
const int n_arm_size = 50;
const auto tol = 1e-8;
const Eigen::Vector2d thresholds{1.96, 1.96};

using mat_t = DirectBayesBinomialControlkTreatment::mat_t;
using vec_t = DirectBayesBinomialControlkTreatment::vec_t;

TEST(DirectBayes, TestConditionalExceedProbGivenSigma) {
    mat_t got = DirectBayesBinomialControlkTreatment::
        conditional_exceed_prob_given_sigma(
            1.10517092, 0.1, Eigen::Vector4d{12.32, 10.08, 11.22, 10.08},
            Eigen::Vector4d{0.24116206, -0.94446161, 0.66329422, 0.94446161},
            Eigen::Vector4d{-0.40546511, -0.40546511, -0.40546511, -0.40546511},
            Eigen::Vector4d{0, 0, 0, 0});
    Eigen::Vector4d want;
    want << 2.300336159211112, -1.5088377933958017, 3.55492998559061,
        4.220022558081819;
    EXPECT_TRUE(got.isApprox(want, tol));
    got = DirectBayesBinomialControlkTreatment::
        conditional_exceed_prob_given_sigma(
            1.01445965e-8, 0.1, Eigen::Vector4d{12.32, 10.08, 11.22, 10.08},
            Eigen::Vector4d{0.24116206, -0.94446161, 0.66329422, 0.94446161},
            Eigen::Vector4d{-0.40546511, -0.40546511, -0.40546511, -0.40546511},
            Eigen::Vector4d{0, 0, 0, 0});
    want << 4.39227936, 4.39227836, 4.39227966, 4.39227978;
    EXPECT_TRUE(got.isApprox(want, tol));
};

TEST(DirectBayes, TestGetFalseRejections) {
    const auto [quadrature_points, weighted_density_logspace] =
        DirectBayesBinomialControlkTreatment::get_quadrature(
            alpha_prior, beta_prior, n_points, n_arm_size);
    const vec_t phat = Eigen::Vector4d{28, 14, 33, 36}.array() / 50;
    const auto got = DirectBayesBinomialControlkTreatment::get_false_rejections(
        phat, quadrature_points, weighted_density_logspace, thresholds,
        n_arm_size, mu_sig_sq);
    Eigen::Vector4d want{-9.08741083467441, -10.771399777758091,
                         -8.169764419088231, -7.569830004638335};
    EXPECT_TRUE(got.isApprox(want, tol));
};

TEST(DirectBayes, TestFastInvert) {
    auto S = Eigen::Matrix4d();
    S << 1.2051709180756478, 0.1, 0.1, 0.1, 0.1, 1.2051709180756478, 0.1, 0.1,
        0.1, 0.1, 1.2051709180756478, 0.1, 0.1, 0.1, 0.1, 1.2051709180756478;
    auto d = Eigen::Vector4d();
    d << 12.32, 10.08, 11.219999999999999, 10.080000000000002;
    const auto got = DirectBayesBinomialControlkTreatment::fast_invert(S, d);
    auto want = Eigen::Matrix4d();
    want << 0.07596619108756637, 0.00042244834619945633, 0.0003827289587913419,
        0.000422448346199456, 0.00042244834619945633, 0.09154316740682344,
        0.00046077407089149705, 0.0005085929343690617, 0.0003827289587913419,
        0.00046077407089149705, 0.08289278392798395, 0.00046077407089149727,
        0.000422448346199456, 0.0005085929343690617, 0.00046077407089149727,
        0.09154316740682344;
    EXPECT_TRUE(want.isApprox(got, tol));
};
}  // namespace
}  // namespace kevlar