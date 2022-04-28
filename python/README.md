# PyKevlar

## Dependencies

The full list of dependencies can be seen in the [`environment.yml` file](../environment.yml).

The most important dependencies are:
- [conda](https://docs.conda.io/projects/conda/en/latest/index.html)
    - [Anaconda](https://www.anaconda.com/)
    - [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Python >= 3.9](https://www.python.org/)
- [pybind11](https://pybind11.readthedocs.io/en/stable/)

## Install

Please follow the directions in the [main repo README](../README.md) for setting up your development environment including your conda environment:

See [Dependencies](#dependencies) for the full list of dependencies. These should already be installed via the conda environment you have set up.

```
bazel build --config clang //python:pykevlar_wheel
pip install bazel-bin/python/dist/pykevlar-0.1-py3-none-any.whl
```

## Reinstall

To reinstall the top-level Python package,
run the following:
```
bazel build --config clang //python:pykevlar_wheel
pip install --force-reinstall bazel-bin/python/dist/pykevlar-0.1-py3-none-any.whl
```

## Smoke test

```
bazel run -c opt //python/example:simple_selection -- main
```
