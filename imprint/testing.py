"""
Here you will find tools for snapshot testing. Snapshot testing is a way to
check that the output of a function is the same as it used to be. This is
particularly useful for end to end tests where we don't have a comparison point
for the end result but we want to know when the result changes. Snapshot
testing is very common in numerical computing.

Usage example:

```
def test_foo(snapshot):
    K = 8000
    result = scipy.stats.binom.std(n=K, p=np.linspace(0.4, 0.6, 100)) / K
    np.testing.assert_allclose(result, snapshot(result))
```

If you run `pytest --snapshot-update test_file.py::test_foo`, the snapshot will
be saved to disk. Then later when you run `pytest test_file.py::test_foo`, the
`snapshot(...)` call will automatically load that object so that you can
compare against the loaded object.

It's fine to call `snapshot(...)` multiple times in a test. The snapshot
filename will have an incremented counter indicating which call index is next.

When debugging a snapshot test, you can directly view the snapshot file if you
are using the `TextSerializer`. This is the default. Pandas DataFrame objects
are saved as csv and numpy arrays are saved as txt files.
"""
import glob
import os
import pickle
from pathlib import Path

import jax.numpy
import numpy as np
import pandas as pd
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """
    Exposes snapshot plugin configuration to pytest.
    https://docs.pytest.org/en/latest/reference.html#_pytest.hookspec.pytest_addoption
    """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        dest="update_snapshots",
        help="Update snapshots",
    )


def path_and_check(filebase, ext):
    snapshot_path = filebase + "." + ext
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(
            f"Snapshot file not found: {snapshot_path}."
            " Did you forget to run with --snapshot-update?"
        )
    return snapshot_path


class Pickler:
    @staticmethod
    def serialize(filebase, obj):
        with open(filebase + ".pkl", "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def deserialize(filebase, obj):
        with open(path_and_check(filebase, "pkl"), "rb") as f:
            return pickle.load(f)


class TextSerializer:
    @staticmethod
    def serialize(filebase, obj):
        if isinstance(obj, pd.DataFrame):
            # in all our dataframes, the index is meaningless, so we do not
            # save it here.
            obj.to_csv(filebase + ".csv", index=False)
        elif isinstance(obj, np.ndarray) or isinstance(obj, jax.numpy.DeviceArray):
            np.savetxt(filebase + ".txt", obj)
        elif np.isscalar(obj):
            np.savetxt(filebase + ".txt", np.array([obj]))
        else:
            raise ValueError(
                f"TextSerializer cannot serialize {type(obj)}."
                " Try calling snapshot(obj, serializer=Pickler)."
            )

    @staticmethod
    def deserialize(filebase, obj):
        if isinstance(obj, pd.DataFrame):
            return pd.read_csv(path_and_check(filebase, "csv"))
        elif isinstance(obj, np.ndarray) or isinstance(obj, jax.numpy.DeviceArray):
            return np.loadtxt(path_and_check(filebase, "txt"))
        elif np.isscalar(obj):
            return np.loadtxt(path_and_check(filebase, "txt"))
        else:
            raise ValueError(
                f"TextSerializer cannot deserialize {type(obj)}."
                " Try calling snapshot(obj, serializer=Pickler)."
            )


class SnapshotAssertion:
    def __init__(
        self,
        *,
        update_snapshots,
        request,
        default_serializer=TextSerializer,
    ):
        self.update_snapshots = update_snapshots
        self.clear_snapshots = update_snapshots
        self.request = request
        self.default_serializer = default_serializer
        self.calls = 0

    def _get_filebase(self):
        test_folder = Path(self.request.fspath).parent
        test_name = self.request.node.name
        return test_folder.joinpath("__snapshot__", test_name + f"_{self.calls}")

    def get(self, obj, serializer=None):
        if serializer is None:
            serializer = self.default_serializer

        return serializer.deserialize(str(self._get_filebase()), obj)

    def __call__(self, obj, serializer=None):
        """
        Return the saved copy of the object. If --snapshot-update is passed,
        save the object to disk in the __snapshot__ folder.

        Args:
            obj: The object to compare against. This is needed here to
                 determine the file extension.
            serializer: The serializer for loading the snapshot. Defaults to
                None which means we will use default_serializer. Unless
                default_serializer has been changed, this is TextSerializer, which
                will save the object as a .txt or .csv depending on whether it's a
                pd.DataFrame or np.ndarray.

        Returns:
            The snapshotted object.
        """
        if serializer is None:
            serializer = self.default_serializer

        # We provide the serializer with a filename without an extension. The
        # serializer can choose what extension to use.
        filebase = self._get_filebase()
        self.calls += 1
        if self.update_snapshots:
            filebase.parent.mkdir(exist_ok=True)
            str_filebase = str(filebase)
            # Delete any existing snapshots with the same name and index
            # regardless of the file extension.
            delete_files = glob.glob(str_filebase + ".*")
            for f in delete_files:
                os.remove(f)
            serializer.serialize(str_filebase, obj)
        return serializer.deserialize(str(filebase), obj)


@pytest.fixture
def snapshot(request):
    return SnapshotAssertion(
        update_snapshots=request.config.option.update_snapshots,
        request=request,
    )
