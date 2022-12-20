import time
from pathlib import Path

import IPython
import matplotlib
import pandas as pd


def run_tutorial(filename):
    matplotlib.use("Agg")
    ipy = IPython.terminal.embed.InteractiveShellEmbed()
    path = Path(__file__).resolve().parent.parent.joinpath("tutorials", filename)
    start = time.time()
    ipy.run_line_magic("run", str(path))
    end = time.time()
    return ipy.user_ns, end - start


def test_ztest_tutorial(snapshot):
    nb_namespace, _ = run_tutorial("ztest.ipynb")
    rej_df = nb_namespace["rej_df"]
    pd.testing.assert_frame_equal(rej_df, snapshot(rej_df))


if __name__ == "__main__":
    from unittest import mock
    import matplotlib.pyplot

    with mock.patch("matplotlib.pyplot"):
        for i in range(3):
            ns, runtime = run_tutorial("ztest.ipynb")
            print(runtime)
