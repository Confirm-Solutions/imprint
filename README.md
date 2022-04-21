# Kevlar

The Kevlar Project.

## Dependencies

- [pre-commit](https://pre-commit.com/)

## Install

Please run all the steps here to get a fully functional development environment.

If you do not have conda installed already, please install it. There are many
ways to get conda. We recommend installing `Mambaforge` which is a conda
installation wwith `mamba` installed by default and set to use `conda-forge` as
the default set of package repositories. [Click here for installers and
installation instructions.](https://github.com/conda-forge/miniforge#mambaforge)

To clone the git repo:
```
git clone git@github.com:mikesklar/kevlar.git
```

To set up your kevlar conda environment (note that you may substitute `mamba`
here for `conda` and the install will be substantially faster):
```
cd kevlar/
conda update conda
conda env create
conda activate kevlar
```

To set up pre-commit:
```
pre-commit install
```

To set up your bazel configuration for building C++:
```
./generate_bazelrc
```

From here, we refer to the installation instructions
for each of the sub-components:

- [PyKevlar](./python/README.md): Kevlar Python package.
- [kevlar](./kevlar/README.md): Kevlar C++ core engine.
