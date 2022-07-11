import argparse
import os
from datetime import timedelta
from logging import basicConfig
from logging import DEBUG as log_level
from logging import getLogger
from logging import WARNING
from timeit import default_timer as timer

import numpy as np
from pyimprint.driver import accumulate_process
from pyimprint.grid import HyperPlane
from pyimprint.grid import make_cartesian_grid_range
from pyimprint.model.binomial import Thompson
from utils import to_array


def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def logit(p):
    return np.log(p / (1 - p))


# ==========================================

sims_def = int(1e5)
points_def = 128
threads_def = os.cpu_count()
samples_def = 100
alpha_prior_def = 1.0
beta_prior_def = 1.0
p_thresh_def = sigmoid(0.6)
seed_def = 69
critval_def = 0.95
lower_def = logit(0.4)
upper_def = logit(0.8)
delta_def = 0.01
bound_def = False
hsh_def = ""
iters_def = 15
max_sims_def = int(1e5)
max_batch_def = int(1e6)
init_sims_def = int(1e3)
init_points_def = 8
alpha_def = 0.05
critval_tol_def = alpha_def * 1.1
do_plot_def = False

common_parser = argparse.ArgumentParser(
    description="Common parser.",
    add_help=False,
)
common_parser.add_argument(
    "--samples",
    type=int,
    nargs="?",
    default=samples_def,
    help=f"Number of samples in each arm (default: {samples_def}).",
)
common_parser.add_argument(
    "--alpha_prior",
    type=float,
    nargs="?",
    default=alpha_prior_def,
    help=f"Alpha prior (default: {alpha_prior_def}).",
)
common_parser.add_argument(
    "--beta_prior",
    type=float,
    nargs="?",
    default=beta_prior_def,
    help=f"Beta prior (default: {alpha_prior_def}).",
)
common_parser.add_argument(
    "--p_thresh",
    type=float,
    nargs="?",
    default=p_thresh_def,
    help=f"Posterior exceedance threshold (default: {p_thresh_def}).",
)
common_parser.add_argument(
    "--seed",
    type=int,
    nargs="?",
    default=seed_def,
    help=f"Number of samples in each arm (default: {seed_def}).",
)
common_parser.add_argument(
    "--lower",
    type=float,
    nargs="*",
    default=lower_def,
    help=f"Lower bound of grid-points along each dimension (default: {lower_def}). "
    "Must be either length 1 or same as --arms.",
)
common_parser.add_argument(
    "--upper",
    type=float,
    nargs="*",
    default=upper_def,
    help=f"Upper bound of grid-points along each dimension (default: {upper_def}). "
    "Must be either length 1 or same as --arms.",
)
common_parser.add_argument(
    "--delta",
    type=float,
    nargs="?",
    default=delta_def,
    help=f"Imprint bound 1-confidence (default: {delta_def}).",
)

global_parser = argparse.ArgumentParser(
    description="""
    Example of simulating a binomial simple selection model.
    """,
)
sub_parsers = global_parser.add_subparsers(
    dest="example_type",
    help="Types of examples.",
    required=True,
)

main_parser = sub_parsers.add_parser(
    "main",
    parents=[common_parser],
    help="Main example parser.",
)
main_parser.add_argument(
    "--sims",
    type=int,
    nargs="?",
    default=sims_def,
    help=f"Number of total simulations (default: {sims_def}).",
)
main_parser.add_argument(
    "--points",
    type=int,
    nargs="?",
    default=points_def,
    help="Number of evenly spaced out points along one dimension "
    f"(default: {points_def}). "
    "The generated points will form a cartesian product "
    "with dimension specified by --arms.",
)
main_parser.add_argument(
    "--threads",
    type=int,
    nargs="?",
    default=threads_def,
    help=f"Number of threads (default: {threads_def}).",
)
main_parser.add_argument(
    "--critval",
    type=float,
    nargs="?",
    default=critval_def,
    help=f"Critical value for test rejection (default: {critval_def}).",
)
main_parser.add_argument(
    "--bound",
    action="store_const",
    const=(not bound_def),
    default=bound_def,
    help=f"Computes imprint bound with level --delta if True (default: {bound_def}).",
)
main_parser.add_argument(
    "--hash",
    type=str,
    nargs="?",
    default=hsh_def,
    help=f"Hash to append to imprint bound output (default: {hsh_def}).",
)

adagrid_parser = sub_parsers.add_parser(
    "adagrid",
    parents=[common_parser],
    help="AdaGrid example.",
)
adagrid_parser.add_argument(
    "--iters",
    type=int,
    nargs="?",
    default=iters_def,
    help=f"Runs adagrid with this number of max iterations (default: {iters_def}).",
)
adagrid_parser.add_argument(
    "--max_sims",
    type=int,
    nargs="?",
    default=max_sims_def,
    help="Runs adagrid with this number of "
    f"max simulation size (default: {max_sims_def}).",
)
adagrid_parser.add_argument(
    "--max_batch",
    type=int,
    nargs="?",
    default=max_batch_def,
    help="Runs adagrid with this number of max "
    f"grid-point batch size (default: {max_batch_def}).",
)
adagrid_parser.add_argument(
    "--init_sims",
    type=int,
    nargs="?",
    default=init_sims_def,
    help="Runs adagrid with this number of "
    f"initial simulation size (default: {init_sims_def}).",
)
adagrid_parser.add_argument(
    "--init_points",
    type=int,
    nargs="?",
    default=init_points_def,
    help="Runs adagrid with this number of "
    f"initial grid-points along each direction (default: {init_points_def}).",
)
adagrid_parser.add_argument(
    "--alpha",
    type=float,
    nargs="?",
    default=alpha_def,
    help=f"Runs adagrid with test target nominal level alpha (default: {alpha_def}).",
)
adagrid_parser.add_argument(
    "--critval_tol",
    type=float,
    nargs="?",
    default=critval_tol_def,
    help=f"""
    Runs adagrid with grid-point finalize condition (default: {critval_tol_def}).
    If a grid-point has estimated nominal level < to this value,
    adagrid does not operate on that grid-point anymore.
    The higher the value, the more quickly adagrid will finish,
    but more likely the points will not have a good configuration.
    """,
)
adagrid_parser.add_argument(
    "--plot",
    action="store_const",
    const=(not do_plot_def),
    default=do_plot_def,
    help=f"Plots AdaGrid results if True (default: {do_plot_def}).",
)

args = global_parser.parse_args()

# outer args
n_arms = 2
alpha_prior = args.alpha_prior
beta_prior = args.beta_prior
p_thresh = args.p_thresh
seed = args.seed
lower = args.lower
upper = args.upper
delta = args.delta
n_samples = args.samples

lower = to_array(lower, n_arms)
upper = to_array(upper, n_arms)

# main example args
if args.example_type == "main":
    sim_size = args.sims
    n_thetas_1d = args.points
    n_threads = args.threads
    critval = args.critval
    bound = args.bound
    hsh = args.hash
    if bound:
        from utils import create_ub_plot_inputs, save_ub

# adagrid args
elif args.example_type == "adagrid":
    n_iter = args.iters
    N_max = args.max_sims
    max_batch_size = args.max_batch
    init_sim_size = args.init_sims
    init_size = args.init_points
    alpha = args.alpha
    finalize = args.critval_tol
    do_plot = args.plot

    # imports conditional on command-line args
    from pyimprint.batcher import SimpleBatch
    from pyimprint.grid import AdaGrid
    from scipy.stats import norm

    # Disable matplotlib logging
    getLogger("matplotlib").setLevel(WARNING)
    import matplotlib.pyplot as plt

# Begin our logging
basicConfig(
    level=log_level,
    format="%(asctime)s %(levelname)-8s %(module)-20s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = getLogger(__name__)
logger.info("n_arms: {}".format(n_arms))
logger.info("alpha_prior: {}".format(alpha_prior))
logger.info("beta_prior: {}".format(beta_prior))
logger.info("p_thresh: {}".format(p_thresh))
logger.info("n_samples: {}".format(n_samples))
logger.info("seed: {}".format(seed))
logger.info("lower: {}".format(lower))
logger.info("upper: {}".format(upper))
logger.info("delta: {}".format(delta))

if args.example_type == "main":
    logger.info("sim_size: {}".format(sim_size))
    logger.info("n_thetas_1d: {}".format(n_thetas_1d))
    logger.info("n_threads: {}".format(n_threads))
    logger.info("critval: {}".format(critval))
    logger.info("bound: {}".format(bound))
    logger.info("hash: {}".format(hsh))

elif args.example_type == "adagrid":
    logger.info("n_iter: {}".format(n_iter))
    logger.info("N_max: {}".format(N_max))
    logger.info("max_batch_size: {}".format(max_batch_size))
    logger.info("init_sim_size: {}".format(init_sim_size))
    logger.info("init_size: {}".format(init_size))
    logger.info("alpha: {}".format(alpha))
    logger.info("finalize: {}".format(finalize))
    logger.info("do_plot: {}".format(do_plot))

# ==========================================

## Begin example code

# set numpy random seed
np.random.seed(seed)

# define null hypos
null_hypos = []
for i in range(n_arms):
    n = np.zeros(n_arms)
    n[i] = -1
    null_hypos.append(HyperPlane(n, -logit(p_thresh)))

# Create current batch of grid points.
# At the process-level, we only need to know theta, radii.

# These parameters are only needed to unify the
# making of cartesian grid range.
grid_n_thetas_1d = None
grid_sim_size = None

if args.example_type == "main":
    grid_n_thetas_1d = n_thetas_1d
    grid_sim_size = sim_size

elif args.example_type == "adagrid":
    grid_n_thetas_1d = init_size
    grid_sim_size = init_sim_size

gr = make_cartesian_grid_range(
    grid_n_thetas_1d,
    lower,
    upper,
    grid_sim_size,
)

# create model
model = Thompson(n_samples, alpha_prior, beta_prior, p_thresh, [])

if args.example_type == "adagrid":
    # TODO: temporary values we feed - needs to change once
    # adagrid becomes more general.
    alpha_minus = alpha - 2 * np.sqrt(alpha * (1 - alpha) / init_sim_size)
    thr = norm.isf(alpha)
    thr_minus = norm.isf(alpha_minus)

    # create batcher
    batcher = SimpleBatch(max_size=max_batch_size)
    adagrid = AdaGrid()
    gr_new = adagrid.fit(
        batcher=batcher,
        model=model,
        null_hypos=null_hypos,
        init_grid=gr,
        alpha=alpha,
        delta=delta,
        seed=seed,
        max_iter=n_iter,
        N_max=N_max,
        alpha_minus=alpha_minus,
        thr=thr,
        thr_minus=thr_minus,
        finalize_thr=finalize,
        rand_iter=False,
        debug=True,
    )

    finals = None
    curr = None

    # iterate through adagrid and study output
    i = 0
    adagrid_time = 0
    while 1:
        try:
            start = timer()
            curr, finals = next(gr_new)
            end = timer()
            adagrid_time += end - start
        except StopIteration:
            break

        if do_plot:
            thetas = curr.thetas()

            plt.scatter(
                thetas[0, :],
                thetas[1, :],
                marker=".",
                c=curr.sim_sizes(),
                cmap="plasma",
            )

            plt.show()
        i += 1

    n_pts = 0
    s_max = 0
    if not (curr is None):
        finals.append(curr)
    for final in finals:
        n_pts += final.thetas().shape[1]
        if final.sim_sizes().size != 0:
            s_max = max(s_max, np.max(final.sim_sizes()))

    logger.info("AdaGrid n_gridpts: {}".format(n_pts))
    logger.info("AdaGrid max_sim_size: {}".format(s_max))
    logger.info("AdaGrid n_iters: {}".format(i))
    logger.info("AdaGrid time: {}".format(timedelta(seconds=adagrid_time)))

elif args.example_type == "main":
    model.critical_values([critval])

    gr.create_tiles(null_hypos)

    start = timer()
    gr.prune()
    end = timer()

    logger.info("Prune time: {}".format(timedelta(seconds=end - start)))
    logger.info("n_gridpts: {}".format(gr.n_gridpts()))
    logger.info("n_tiles: {}".format(gr.n_tiles()))

    start = timer()
    out = accumulate_process(model, gr, sim_size, seed, n_threads)
    end = timer()

    logger.info("Accumulate time: {}".format(timedelta(seconds=end - start)))

    # create upper bound plot inputs and save info
    if bound:
        start = timer()
        P, B = create_ub_plot_inputs(model, out, gr, delta)
        end = timer()
        logger.info("Create plot input time: {}".format(timedelta(seconds=end - start)))

        suffix = "thompson"
        if hsh != "":
            suffix += "-" + hsh

        start = timer()
        save_ub(
            f"P-{suffix}.csv",
            f"B-{suffix}.csv",
            P,
            B,
        )
        end = timer()
        logger.info("CSV write time: {}".format(timedelta(seconds=end - start)))

    # print type I error
    logger.info("Type I error: {}".format(out.typeI_sum() / sim_size))
