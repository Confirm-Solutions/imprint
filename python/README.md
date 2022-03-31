# PyKevlar

## Dependencies

- [conda](https://docs.conda.io/projects/conda/en/latest/index.html)
    - [Anaconda](https://www.anaconda.com/)
    - [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Python >= 3.9](https://www.python.org/)
- [pybind11](https://pybind11.readthedocs.io/en/stable/)

## Install

We recommend using a `conda` environment for development.
The following is an example script to create an environment:
```
conda update conda
conda create -n kevlar python=3.9.7 anaconda
conda activate kevlar
```

If one wishes to have a local installation of the Python 
package dependencies such as `pybind11`, simply run:
```
conda install pybind11
```
See [Dependencies](#dependencies) for the full list of dependencies.

```
pip install -r requirements.txt
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

TODO: we should probably make a shell script wrapping that.

## Smoke test

```
bazel run --config clang -c opt //python:fit_driver_example
```
