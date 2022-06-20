from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython

# this let's me leave in the "@profile" lines when I'm not running
# line_profiler.
if "profile" not in __builtins__:
    __builtins__["profile"] = lambda x: x


@dataclass
class QuadRule:
    pts: np.ndarray
    wts: np.ndarray


def broadcast(arr, target_shape, dest_dims):
    """
    This is a helper functions for the gnarly broadcasting operations required
    in the quadrature code.

    arr: the array to broadcast
    target_shape: the requested broadcast shape.
    dest_dims: the index of the output dimensions corresponding to each input dimension.
    """

    broadcast_shape = [1] * len(target_shape)
    for d, size in enumerate(arr.shape):
        target_dim = dest_dims[d]
        broadcast_shape[target_dim] = size
        if not target_shape[target_dim] == size:
            raise ValueError(
                f"Input array size of {size} along dimension {d}"
                f" does not match requested shape of {target_shape[d]}"
                f" on dimension {target_dim}."
            )
    return arr.reshape(broadcast_shape)


def simpson_rule(n, a=-1, b=1):
    """
    Output the points and weights for a Simpson rule quadrature on the interval
    (a, b)
    """
    if not (n >= 3 and n % 2 == 1):
        raise ValueError("Simpson's rule is only defined for odd n >= 3.")
    h = (b - a) / (n - 1)
    pts = np.linspace(a, b, n)
    wts = np.empty(n)
    wts[0] = 1
    wts[1::2] = 4
    wts[2::2] = 2
    wts[-1] = 1
    wts *= h / 3
    return QuadRule(pts, wts)


def composite_rule(q_rule_fnc, *domains):
    pts = []
    wts = []
    for d in domains:
        qr = q_rule_fnc(*d)
        pts.append(qr.pts)
        wts.append(qr.wts)
    pts = np.concatenate(pts)
    wts = np.concatenate(wts)
    return QuadRule(pts, wts)


def gauss_rule(n, a=-1, b=1):
    """
    Points and weights for a Gaussian quadrature with n points on the interval
    (a, b)
    """
    pts, wts = np.polynomial.legendre.leggauss(n)
    pts = (pts + 1) * (b - a) / 2 + a
    wts = wts * (b - a) / 2
    return QuadRule(pts, wts)


def log_gauss_rule(N, a, b):
    A = np.log(a)
    B = np.log(b)
    qr = gauss_rule(N, a=A, b=B)
    pts = np.exp(qr.pts)
    wts = np.exp(qr.pts) * qr.wts
    return QuadRule(pts, wts)


def integrate_multidim(f, axes, quad_rules):
    """
    Integrate a function along the specified array dimensions using the
    specified quadrature rules.

    e.g.
    integrate_multidim(f, (2,1), (gauss_rule(5), gauss_rule(6)))
    where f is an array with shape (2, 6, 5, 3)

    will perform a multidimensional integral along the second and third axes
    resulting in an output with shape (2, 3)
    """
    If = f
    # integrate the last axis first so that the axis numbers don't change.
    reverse_axes = np.argsort(axes)[::-1]
    for idx in reverse_axes:
        ax = axes[idx]
        q = quad_rules[idx]

        # we need to make sure q has the same dimensionality as If to make numpy
        # happy. this converts the weight array from shape
        # e.g. (15,) to (1, 1, 15, 1)
        broadcast_shape = [1] * If.ndim
        broadcast_shape[ax] = q.pts.shape[0]
        q_broadcast = q.wts.reshape(broadcast_shape)

        # actually integrate!
        If = np.sum(q_broadcast * If, axis=ax)
    return If


### Tools for setting up nice Jupyter notebooks.
def magic(text):
    ipy = get_ipython()
    if ipy is not None:
        ipy.magic(text)


def configure_mpl_fast():
    """
    No retina and no latex, fast matplotlib. This can be useful for making
    lots of complex plots. The "pretty" plots often take 5x longer.
    """
    magic("config InlineBackend.figure_format='png'")


def configure_mpl_pretty():
    """Retina and Latex matplotlib figures"""
    magic("config InlineBackend.figure_format='retina'")
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def setup_nb(text_size_ratio=0.75, pretty=True, autoreload=True):
    """
    This function is handy to call at the top of a Jupyter notebook. It sets up:
    1. autoreload for allowing python modules to be modified without restart the
       notebook.
    2. sane matplotlib defaults for making good looking figures including:
        - retina mode
        - good colors
        - solid non-transparent background
        - nice text sizes
    """
    if autoreload:
        magic("load_ext autoreload")
        magic("autoreload 2")

    if pretty:
        configure_mpl_pretty()
    else:
        configure_mpl_fast()

    plt.rcParams["axes.facecolor"] = (1.0, 1.0, 1.0, 1.0)
    plt.rcParams["figure.facecolor"] = (1.0, 1.0, 1.0, 1.0)
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["image.cmap"] = "plasma"
    scale_text(factor=text_size_ratio)
    np.set_printoptions(edgeitems=10, linewidth=100)


def scale_text(factor=0.75):
    plt.rcParams["font.size"] = 20 * factor
    plt.rcParams["axes.labelsize"] = 18 * factor
    plt.rcParams["axes.titlesize"] = 20 * factor
    plt.rcParams["xtick.labelsize"] = 16 * factor
    plt.rcParams["ytick.labelsize"] = 16 * factor
    plt.rcParams["legend.fontsize"] = 20 * factor
    plt.rcParams["figure.titlesize"] = 22 * factor
