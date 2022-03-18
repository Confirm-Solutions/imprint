# Kevlar

The Kevlar Project.

## Dependencies

- [conda](https://docs.conda.io/projects/conda/en/latest/index.html)
- [Python >= 3.9](https://www.python.org/)
- [pybind11](https://pybind11.readthedocs.io/en/stable/)
- [pre-commit](https://pre-commit.com/)

## Installation

We recommend using a `conda` environment for development.
The following is an example script to create an environment:
```
conda update conda
conda create -n kevlar python=3.9.7 anaconda
conda activate kevlar
```

If one wishes to have a local installation of the Python 
package dependencies such as `pybind11`, 
simply run:
```
conda install pybind11
```
See [Dependencies](#dependencies) for the full list of dependencies.

Next, clone the repo:
```
git clone git@github.com:mikesklar/kevlar.git
cd kevlar/
pre-commit install
```
We refer to the [C++ Core Engine README](src/kevlar/README.md) 
to install the C++ core engine dependencies.

Finally, we install the top-level Python package
in editable mode (`-e`) for development:
```
cd python/
pip install -r requirements.txt
pip install -vvv -e .
```

## Reinstall

To reinstall the top-level Python package,
run the following:
```
cd python/
rm -rf ./build ./dist ./*egg-info
find . -type f -name "*.so" -exec rm {} \;
find . -type f -name "*.o" -exec rm {} \;
pip install -vvv -e .
```

TODO: we should probably make a shell script wrapping that.

A simple, hacky method is to first `touch` any `.cpp` file
in `python/src/`, as our setup script will automatically
rebuild all source files, then install as usual.
An example script is the following:
```
cd python/
touch src/core.cpp
pip install -vvv -e .
```

## Smoke test
```
cd python/
PYTHONPATH=. time python ./examples/fit_driver_example.py
```
