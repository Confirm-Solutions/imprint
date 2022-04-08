from dataclasses import dataclass
from operator import index
from typing import Callable

import numpy as np
import scipy.stats
import util


@dataclass
class Model:
    """
    A generic description of a probabalistic model with enough detail to run
    INLA.
    """

    log_prior: Callable
    log_joint: Callable
    log_joint_xonly: Callable
    gradx_log_joint: Callable
    hessx_log_joint: Callable


##################
###### EXACT #####
##################
# NOTE: An exact integrator is partially implemented in the berry_exact.ipynb
# notebook

##################
###### INLA ######
##################
"""
For the INLA code, please see the "INLA from Scratch" paper as the best intro to
INLA concepts.

Note that the variable names here are bit different that in the Berry code:
- $\theta$ in the Berry code is called "x" here to match with typical INLA notation.
- I have also referred to the hyperparameters as "hyper" so as not to use the
  "theta" notation from INLA and make the situtation even more confusing.
"""


def optimize_x0(model, data, hyper):
    """
    Calculate the maximum ("mode") of p(x, y, hyper) holding y, hyper
    fixed.

    Returns:
        minimizer: A dictionary with the same entries as
            scipy.optimize.minimize. Access minimizer['x'] to get the value of the
            minimizer.
    """
    tol = 1e-8
    max_iter = 500
    n_sims = data.shape[0]
    n_hyper = hyper.shape[1]
    n_rows = data.shape[2]
    x = np.zeros((n_sims, n_hyper, n_rows))

    status = 0
    success = False
    message = "Success"
    for i in range(max_iter):
        fj = model.gradx_log_joint(x, data, hyper)
        fh = model.hessx_log_joint(x, data, hyper)
        update = -fj / fh
        x += update

        # What is the correct stopping criterion here?
        # based on changes in solution?
        # based on changes in objective value?
        # based on smaller values in the jacobian?
        # what does R-INLA do?
        # Currently, I'm checking that changes in the solution converge to a
        # small step size.
        if np.max(np.linalg.norm(update, axis=-1)) < tol:
            break
        if i == max_iter - 1:
            status = 1
            success = False
            message = "Reached max_iter without converging."

    f = model.log_joint_xonly(x, data, hyper)

    # NOTE: It might be possible to return the gradient and hessian calculated here as
    # approximations of the true gradient and hessian as the optimum. But, I'm
    # not sure whether the approximation is sufficiently accurate. Because a
    # final optimization step has been taken, the gradient and hessian are one
    # step out of date.
    soln = dict(
        x=x,
        fun=f,
        nfev=1,
        njev=i,
        nhev=i,
        status=status,
        success=success,
        message=message,
    )
    return soln


def calc_log_posterior_hyper(model, data, hyper):
    """
    This function calculates log p(hyper | y): the log posterior of the
    hyperparameters given the data.

    `model` is Model object!
    `data` is expected to have shape: (n_simulations, n_rows, data_dim)
    `hyper` is expected to have shape: (n_hyper, hyper_dim)
    """

    # `hyper` is expected to have shape: (n_hyper1, n_hyper2, ..., hyper_dim)
    # `data` is expected to have shape: (n_simulations, n_rows, data_dim)
    # To nicely broadcast during array operations, we reshape to:
    # hyper_broadcast: (1, n_hyper, hyper_dim)
    # data_broadcast: (n_simulations, 1, n_rows, data_dim)
    hyper_broadcast = hyper.reshape((1, -1, 2))
    data_broadcast = data[:, None, :]

    # Step 1) Find the maximum of the joint distribution with respect to the
    # latent variables, x, while holding data/hyper fixed.
    x0_info = optimize_x0(model, data_broadcast, hyper_broadcast)
    x0 = x0_info["x"]
    # Check to make sure the gradient is actually small!
    grad_check = model.gradx_log_joint(x0, data_broadcast, hyper_broadcast)
    np.testing.assert_allclose(grad_check, 0, atol=1e-5)

    # The INLA approximation reduces to a simple expression! See the INLA
    # from Scratch post or the original INLA paper for a derivation.
    # log p(hyper | y) = log p(y, x_0, hyper) - 0.5 * log (det(-H(y, x_0, hyper)))
    # where H is the hessian at the maximum. Intuitively, this comes from a
    # quadratic approximation the log density at the maximum point. When
    # exponentiated, this is a normal distribution.
    H = model.hessx_log_joint(x0, data_broadcast, hyper_broadcast)
    detnegH = np.prod(-H, axis=-1)
    ljoint = model.log_joint(model, x0, data_broadcast, hyper_broadcast)
    logpost = ljoint - 0.5 * np.log(detnegH)

    # It's handy to return more than just the log posterior since we can re-use
    # some of these intermediate calculations.
    return dict(x0=x0, x0_info=x0_info, H=H, logjoint=ljoint, logpost=logpost)


def calc_posterior_hyper(model, data, quad_rules):
    """
    This function calculates p(hyper | y): the posterior of the hyperparameters
    given the data.

    The basic outline is:
    - choose a grid of hyper values.
    - calculate log p(hyper | y)
    - exponentiate and normalize by numerically integrate in the hyperparameters

    `model` is Model object!
    `data` is expected to have shape: (n_simulations, n_rows, data_dim)
    """

    # Construct a grid of hyper values with shape:
    # (n_hyper1, n_hyper2, ..., n_hyperN, N)
    # For example a two parameter grid of mu/sigma2 might look like:
    # (11, 15, 2) if there were 11 values of mu and 15 values of sigma.
    # hyper_grid[:, :, 0] would be the value of mu at the grid points.
    # hyper_grid[:, :, 1] would be the value of sigma2 at the grid points.
    hyper_grid = np.stack(
        np.meshgrid(*[q.pts for q in quad_rules], indexing="ij"), axis=-1
    )

    logpost_data = calc_log_posterior_hyper(model, data, hyper_grid)
    logpost = logpost_data["logpost"]

    # Exponentiating a large number might result in numerical overflow if the
    # value exceeds the maximum value representable in 64 bit floating point
    # arithmetic.
    #
    # By subtracting (max - 600), we ensure that the largest value in logpost is
    # exactly 600. e^600 is a large but not overflowing exponential so this will
    # make sure the highest parts (most relevant) of the density are well
    # represented.
    logpost -= np.max(logpost, axis=1)[:, None] - 600

    # Exponentiate to get the unnormalized posterior p_u(hyper | y)
    unn_post_hyper = np.exp(logpost).reshape((-1, *hyper_grid.shape[:-1]))

    # Numerically integrate to get the normalization constant. After dividing,
    # post_hyper will be a true PDF.
    normalization_factor = util.integrate_multidim(unn_post_hyper, (1, 2), quad_rules)[
        :, None, None
    ]
    post_hyper = unn_post_hyper / normalization_factor

    # We return the intermediate values from the log posterior calculation and
    # add the hyper grid and quadrature rules to those intermediate values. This
    # is helpful for debugging and reporting.
    report = logpost_data
    report["hyper_grid"] = hyper_grid
    report["hyper_rules"] = quad_rules
    return post_hyper, report


def calc_posterior_x(post_hyper, report, thresh):
    """
    Calculate the marginals of the latent variables, x: p(x_i | y)

    The inputs to this function are exactly the outputs of
    `calc_posterior_hyper`. The approximations used in the construction of the
    hyperparameter posteriors are re-used to calculate latent variable
    marginals. Since INLA assumes latent variable marginals are normally
    distributed, we simply return the mean and std dev of the latent variable
    marginals.
    """
    n_sims = post_hyper.shape[0]
    n_sigma2 = post_hyper.shape[1]
    n_mu = post_hyper.shape[2]
    n_arms = report["x0"].shape[-1]

    x_mu = report["x0"].reshape((n_sims, n_sigma2, n_mu, n_arms))
    H = report["H"]
    x_sigma2 = -(1.0 / H).reshape((n_sims, n_sigma2, n_mu, n_arms))
    x_sigma = np.sqrt(x_sigma2)

    rules = report["hyper_rules"]

    # mu = integral(mu(x | y, hyper) * p(hyper | y))
    mu_post = util.integrate_multidim(x_mu * post_hyper[:, :, :, None], (1, 2), rules)
    T = (x_mu - mu_post[:, None, None, :]) ** 2 + x_sigma2
    var_post = util.integrate_multidim(T * post_hyper[:, :, :, None], (1, 2), rules)
    sigma_post = np.sqrt(var_post)

    # exceedance probabilities
    exceedance = util.integrate_multidim(
        (1.0 - scipy.stats.norm.cdf(thresh[:, None, None, :], x_mu, x_sigma))
        * post_hyper[:, :, :, None],
        (1, 2),
        rules,
    )

    return dict(mu_appx=mu_post, sigma_appx=sigma_post, exceedance=exceedance)


##################
###### ALA #######
##################
# TODO: not implemented
# "Aggressive laplace approximation"

##################
###### MCMC ######
##################
# TODO: this is broken bc I changed the model code


def proposal(x, sigma=0.25):
    rv = scipy.stats.norm.rvs(x, sigma, size=(x.shape[0], 6))

    while np.any(rv[:, 5] < 0):
        # Truncate normal distribution for the precision at 0.
        bad = rv[:, 5] < 0
        badx = x[bad]
        rv[bad, 5] = scipy.stats.norm.rvs(badx[:, 5], sigma, size=badx.shape[0])
    ratio = 1
    return rv, ratio


def mcmc(y, n, iterations=2000, burn_in=500, skip=2):
    def joint(xstar):
        a = xstar[:, -2]
        Qv = xstar[:, -1]
        return np.exp(calc_log_joint(xstar[:, :4], y, n, a, Qv))

    M = y.shape[0]
    x = np.zeros((M, 6))
    x[:, -1] = 1

    Jx = joint(x)
    x_chain = [x]
    J_chain = [Jx]
    accept = [np.ones(M)]

    for i in range(iterations):
        xstar, ratio = proposal(x)

        Jxstar = joint(xstar)
        hastings_ratio = (Jxstar * ratio) / Jx
        U = np.random.uniform(size=M)
        should_accept = U < hastings_ratio
        x[should_accept] = xstar[should_accept]
        Jx[should_accept] = Jxstar[should_accept]

        accept.append(should_accept)
        x_chain.append(x.copy())
        J_chain.append(Jx.copy())
    x_chain = np.array(x_chain)
    J_chain = np.array(J_chain)
    accept = np.array(accept).T

    x_chain_burnin = x_chain[burn_in::skip]

    ci025n = int(x_chain_burnin.shape[0] * 0.025)
    ci975n = int(x_chain_burnin.shape[0] * 0.975)
    results = dict(
        CI025=np.empty(x.shape), CI975=np.empty(x.shape), mean=np.empty(x.shape)
    )
    for j in range(6):
        x_sorted = np.sort(x_chain_burnin[:, :, j], axis=0)
        x_mean = x_sorted.mean(axis=0)
        results["CI025"][:, j] = x_sorted[ci025n]
        results["mean"][:, j] = x_mean
        results["CI975"][:, j] = x_sorted[ci975n]
    return results
