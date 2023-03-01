import jax.numpy as jnp
import numpy as np
import pytest

from imprint.batching import batch
from imprint.batching import batch_yield


def test_simple():
    def f(x):
        return x + 1

    batched_f = batch_yield(f, batch_size=2, in_axes=(0,))
    out = list(batched_f(np.array([1, 2, 3, 4])))
    assert len(out) == 2
    np.testing.assert_allclose(out[0][0], np.array([2, 3]))
    assert out[0][1] == 0
    np.testing.assert_allclose(out[1][0], np.array([4, 5]))
    assert out[1][1] == 0


@pytest.mark.parametrize("module", (np, jnp))
def test_pad(module):
    inputs = module.array([1, 2, 3, 4])

    def f(x):
        assert type(x) == type(inputs)
        return x + 1

    batched_f = batch_yield(f, batch_size=3, in_axes=(0,))
    out = list(batched_f(inputs))
    assert type(out[0][0]) == type(inputs)
    np.testing.assert_allclose(out[1][0], np.array([5, 5, 5]))
    assert out[1][1] == 2


@pytest.mark.parametrize("module", (np, jnp))
def test_multidim(module):
    def f(x):
        return (x.sum(axis=1), x.prod(axis=1))

    for d in range(1, 15):
        inputs = np.random.rand(d, 5)
        inputs = module.array(inputs)
        batched_f = batch(f, batch_size=5, in_axes=(0,))
        out = batched_f(inputs)
        assert type(out[0]) == type(inputs)
        np.testing.assert_allclose(out[0], inputs.sum(axis=1))
        np.testing.assert_allclose(out[1], inputs.prod(axis=1))


def test_multidim_single():
    def f(x):
        return x.sum(axis=1)

    inputs = np.random.rand(7, 5)
    batched_f = batch(f, batch_size=5, in_axes=(0,))
    out = batched_f(inputs)
    np.testing.assert_allclose(out, inputs.sum(axis=1))


def test_out_axes1():
    def f(x):
        return x.T

    inputs = np.random.rand(7, 5)
    batched_f = batch(f, batch_size=5, in_axes=(0,), out_axes=(1,))
    out = batched_f(inputs)
    np.testing.assert_allclose(out, inputs.T)


# NOTE: this doesn't work! make sure we're not doing this anywhere. i think we
# might be...
# def test_out_new_axis():
#     def f(x):
#         return x.sum(axis=1)

#     inputs = np.random.rand(6, 7)
#     batched_f = batch(f, batch_size=5, in_axes=(1,), out_axes=(1,))
#     out = batched_f(inputs)
#     np.testing.assert_allclose(out[:,0] + out[:,1], inputs.sum(axis=1))
