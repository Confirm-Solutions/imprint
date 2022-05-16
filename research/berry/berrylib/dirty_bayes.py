import numpy as np
import scipy.stats
from scipy.special import logit


def fast_invert(S, d):
    for k in range(len(d)):
        offset = (d[k] / (1 + d[k] * S[k, k])) * np.outer(
            S[k],
            S[..., k],
        )
        S = S - offset
    return S


def calc_posterior_x(sigma_sq: float, mu_sig_sq: float, sample_I, thetahat, mu_0, d):
    assert len(sigma_sq) == 1, sigma_sq
    S_0 = np.diag(np.repeat(sigma_sq, d)) + mu_sig_sq
    # V_0 = solve(S_0) #but because this is a known case of the form aI + bJ, we
    # can use the explicit inverse formula, given by: 1/a I - J*(b/(a(a+db)))
    V_0 = np.diag(np.repeat(1 / sigma_sq, d)) - (mu_sig_sq / sigma_sq) / (
        sigma_sq + d * mu_sig_sq
    )
    Sigma_posterior = fast_invert(S_0, sample_I)
    # precision_posterior = V_0 + np.diag(sample_I)
    # Sigma_posterior = np.linalg.inv(precision_posterior)
    mu_posterior = Sigma_posterior @ (sample_I * thetahat + (V_0 * mu_0).sum(axis=-1))
    return mu_posterior, np.diag(Sigma_posterior)


def calc_dirty_bayes(y_i, n_i, mu_0_scalar, logit_p1, thresh, sigma2_rule):
    N, d = y_i.shape
    phat = y_i[:, :] / n_i[:, :]
    # NOTE: we use the logit_p1 offset when converting from p space to logit space.
    thetahat = logit(phat) - logit_p1
    sample_I = n_i[:, :] * phat * (1 - phat)  # diag(n*phat*(1-phat))
    mu_0 = np.full_like(phat, mu_0_scalar)
    mu_sig_sq = 100

    n_sigma2 = sigma2_rule.pts.shape[0]
    mu_posterior = np.empty((N, n_sigma2, d))
    sigma2_posterior = np.empty((N, n_sigma2, d))
    joint_sigma2_y = np.empty((N, n_sigma2))
    for i in range(N):
        for j in range(n_sigma2):
            # Step 1-4: see above.
            sig2 = sigma2_rule.pts[j]
            (mu_posterior[i, j], sigma2_posterior[i, j]) = calc_posterior_x(
                np.array([sig2]),
                np.array([mu_sig_sq]),
                sample_I[i],
                thetahat[i],
                mu_0[i],
                d,
            )
            # Step 6/7
            prior = scipy.stats.invgamma.pdf(sig2, 0.0005, scale=0.000005)
            y_given_sig2 = scipy.stats.multivariate_normal.pdf(
                thetahat[i],
                mu_0[i],
                (np.diag(sample_I[i] ** -1) + np.diag(np.repeat(sig2, d)) + mu_sig_sq),
            )
            joint_sigma2_y[i, j] = prior * y_given_sig2

    # Step 8: Compute the integration weights: p(sigma2 | y) * dsigma2
    py = np.sum(joint_sigma2_y * sigma2_rule.wts[None, :], axis=1)
    sigma2_given_y = joint_sigma2_y / py[:, None]
    weights = sigma2_given_y * sigma2_rule.wts[None, :]

    # Step 9: Compute (mu, sigma) for normal p(theta|y): mixture of gaussians
    mu_db = np.sum(mu_posterior * weights[..., None], axis=1)
    T = (mu_posterior - mu_db[:, None, :]) ** 2 + sigma2_posterior
    sigma2_db = np.sum(T * weights[..., None], axis=1)
    sigma_db = np.sqrt(sigma2_db)

    # Step 10: compute exceedance probability separately.
    exceedance = np.sum(
        (
            1.0
            - scipy.stats.norm.cdf(
                thresh[:, None, :], mu_posterior, np.sqrt(sigma2_posterior)
            )
        )
        * weights[..., None],
        axis=1,
    )

    return dict(
        mu_posterior=mu_posterior,
        sigma2_posterior=sigma2_posterior,
        sigma2_given_y=sigma2_given_y,
        theta_map=mu_db,
        mu_appx=mu_db,
        sigma_appx=sigma_db,
        cilow=mu_db - 2 * sigma_db,
        cihi=mu_db + 2 * sigma_db,
        exceedance=exceedance,
    )
