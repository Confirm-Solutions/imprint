"""
Tools for working with Jupyter notebooks.
1. setup_nb() - call in a notebook to set up nice defaults for plotting.
2. run_tutorial - run a tutorial notebook and return the namespace. Useful for
   testing and benchmarking.

"""
import time
import warnings
from pathlib import Path
from unittest import mock

from imprint import package_settings


def magic(*text):
    from IPython import get_ipython

    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic(*text)


def configure_mpl_fast():
    """
    No retina and no latex, fast matplotlib. This can be useful for making
    lots of complex plots. The "pretty" plots often take 5x longer.
    """
    magic("config", "InlineBackend.figure_format='png'")


def configure_mpl_pretty():
    """Retina and Latex matplotlib figures"""
    import matplotlib.pyplot as plt

    magic("config", "InlineBackend.figure_format='retina'")
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, amssymb}"


def setup_nb(text_size_ratio=1.0, pretty=True, autoreload=True):
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
        magic("load_ext", "autoreload")
        magic("autoreload", "2")

    import matplotlib.pyplot as plt

    if pretty:
        configure_mpl_pretty()
    else:
        configure_mpl_fast()

    plt.rcParams["axes.facecolor"] = (1.0, 1.0, 1.0, 1.0)
    plt.rcParams["figure.facecolor"] = (1.0, 1.0, 1.0, 1.0)
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["image.cmap"] = "plasma"
    # Use the same font for latex and non-latex text
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "STIXGeneral"
    scale_text(factor=text_size_ratio)

    package_settings()


def scale_text(factor=1.0):
    import matplotlib.pyplot as plt

    plt.rcParams["font.size"] = 15 * factor
    plt.rcParams["axes.labelsize"] = 13 * factor
    plt.rcParams["axes.titlesize"] = 15 * factor
    plt.rcParams["xtick.labelsize"] = 12 * factor
    plt.rcParams["ytick.labelsize"] = 12 * factor
    plt.rcParams["legend.fontsize"] = 15 * factor
    plt.rcParams["figure.titlesize"] = 17 * factor


def safe_execfile_ipy(
    self, fname, cell_indices=None, shell_futures=False, raise_exceptions=False
):
    """
    A modification of
    IPython.core.interactiveshell.InteractiveShell.safe_execfile_ipy which adds
    the cell_indices argument. This lets us run only a subset of the cells in
    a notebook.

    Permalink for the copied source. If this function breaks, check if the
    upstream source has changed substantially.
        https://github.com/ipython/ipython/blob/57eaa12cb50c9a95213b9e155032e400b9424871/IPython/core/interactiveshell.py#L2766 # noqa

    Original docstring below:

    Like safe_execfile, but for .ipy or .ipynb files with IPython syntax.
    Parameters
    ----------
    fname : str
        The name of the file to execute.  The filename must have a
        .ipy or .ipynb extension.
    shell_futures : bool (False)
        If True, the code will share future statements with the interactive
        shell. It will both be affected by previous __future__ imports, and
        any __future__ imports in the code will affect the shell. If False,
        __future__ imports are not shared in either direction.
    raise_exceptions : bool (False)
        If True raise exceptions everywhere.  Meant for testing.
    """
    from IPython.core.interactiveshell import prepended_to_syspath
    from IPython.core.interactiveshell import warn

    fname = Path(fname).expanduser().resolve()

    # Make sure we can open the file
    try:
        with fname.open("rb"):
            pass
    except:  # noqa: E722
        warn("Could not open file <%s> for safe execution." % fname)
        return

    # Find things also in current directory.  This is needed to mimic the
    # behavior of running a script from the system command line, where
    # Python inserts the script's directory into sys.path
    dname = str(fname.parent)

    def get_cells():
        """generator for sequence of code blocks to run"""
        if fname.suffix == ".ipynb":
            from nbformat import read

            nb = read(fname, as_version=4)
            if not nb.cells:
                return
            for cell in nb.cells:
                if cell.cell_type == "code":
                    yield cell.source
        else:
            yield fname.read_text(encoding="utf-8")

    with prepended_to_syspath(dname):
        try:
            for i, cell in enumerate(get_cells()):
                if cell_indices is not None and i not in cell_indices:
                    continue
                print(cell)
                result = self.run_cell(cell, silent=True, shell_futures=shell_futures)
                if raise_exceptions:
                    result.raise_error()
                elif not result.success:
                    break
        except:  # noqa: E722
            if raise_exceptions:
                raise
            self.showtraceback()
            warn("Unknown failure executing file: <%s>" % fname)


def run_notebook(filepath, cell_indices=None):
    """
    Programmatically run a notebook return the notebook's namespace and the
    execution time. This is useful for testing notebooks.

    Note: testbook is an alternative to this function. testbook has a critical
    flaw because it currently runs the notebook in a separate process and uses
    JSON to serialize objects. In addition, testbook seems undermaintained and
    I would rather have our own thin 50 LOC implementation.
    https://github.com/nteract/testbook
    see: https://github.com/Confirm-Solutions/confirmasaurus/issues/266

    Args:
        filepath: Path to the notebook to run.
        cell_indices: The indices of the cells to run. Runs all cells if None.
            Defaults to None.

    Returns:
        tuple of (namespace, execution time)
    """
    import IPython
    import matplotlib

    # Using Agg backend to prevent figures from popping up
    matplotlib.use("Agg")

    # mock pyplot because CI doesn't have latex and we don't want to install that.
    with mock.patch("matplotlib.pyplot.show"):
        with mock.patch("matplotlib.pyplot.savefig"):
            ipy = IPython.terminal.embed.InteractiveShellEmbed()
            start = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                safe_execfile_ipy(
                    ipy, filepath, cell_indices=cell_indices, raise_exceptions=True
                )
            end = time.time()
    return ipy.user_ns, end - start
