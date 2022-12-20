from pathlib import Path

import pandas as pd
import pytest

from imprint.nb_util import run_notebook


def get_tutorial_path(filename):
    return Path(__file__).resolve().parent.parent.joinpath("tutorials", filename)


@pytest.mark.slow
def test_ztest_tutorial(snapshot):
    ns, _ = run_notebook(get_tutorial_path("ztest.ipynb"))
    rej_df = ns["rej_df"]
    pd.testing.assert_frame_equal(rej_df, snapshot(rej_df))


@pytest.mark.slow
def test_fisher_exact_tutorial(snapshot):
    ns, _ = run_notebook(
        get_tutorial_path("fisher_exact.ipynb"), cell_indices=[0, 3, 4, 7]
    )
    lamss = ns["cal_df"]["lams"].min()
    assert lamss == snapshot(lamss)


@pytest.mark.slow
def test_basket_tutorial(snapshot):
    ns, _ = run_notebook(get_tutorial_path("basket.ipynb"), cell_indices=[0, 1])
    pd.testing.assert_frame_equal(ns["validation_df"], snapshot(ns["validation_df"]))


def main():
    for i in range(3):
        # ns, runtime = run_notebook(
        #   get_tutorial_path("basket.ipynb"), cell_indices=[0,1])
        ns, runtime = run_notebook(
            get_tutorial_path("fisher_exact.ipynb"), cell_indices=[0, 3, 4, 7]
        )
        # ns, runtime = run_notebook(get_tutorial_path("ztest.ipynb"))
        print(runtime)


if __name__ == "__main__":
    main()
    # main2()
