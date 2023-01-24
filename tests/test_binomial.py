import numpy as np
import pytest

import imprint.bound.binomial as binomial


def test_fwd_binomial_scalar():
    fwd = binomial.BinomialBound.get_forward_bound({"n": 10})
    result = fwd(np.array([0.01]), np.array([0.0]), np.array([[-0.1, 0.1]]))
    np.testing.assert_allclose(result, 0.01530254, rtol=1e-6)


@pytest.mark.parametrize("n", [10, [10, 10]])
def test_fwd_binomial_vector(n):
    fwd = binomial.BinomialBound.get_forward_bound({"n": n})

    f0 = np.array([0.01])
    theta0 = np.array([[0.0, 0.0]])
    vs = np.array([[[-0.1, -0.1], [0.1, 0.1]]])
    result = fwd(f0, theta0, vs)

    np.testing.assert_allclose(result, 0.01870283, rtol=1e-6)
