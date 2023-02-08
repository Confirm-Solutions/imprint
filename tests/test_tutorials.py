from pathlib import Path

import pytest

import imprint as ip
from imprint.nb_util import run_notebook


def get_tutorial_path(filename):
    return Path(__file__).resolve().parent.parent.joinpath("tutorials", filename)


@pytest.mark.slow
def test_ztest_tutorial(snapshot):
    ns, _ = run_notebook(get_tutorial_path("ztest.ipynb"))
    ip.testing.check_imprint_results(ns["g_rej"], snapshot)


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
    ip.testing.check_imprint_results(ns["g_val"], snapshot)


@pytest.mark.slow
def test_chisq_tutorial(snapshot):
    run_notebook(get_tutorial_path("chisq_test.ipynb"))


@pytest.mark.slow
def test_t_test_adaptive_tutorial(snapshot):
    run_notebook(get_tutorial_path("t_test_adaptive.ipynb"))


@pytest.mark.slow
def test_t_test_tutorial(snapshot):
    run_notebook(get_tutorial_path("t_test.ipynb"))


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
