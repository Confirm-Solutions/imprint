# Kevlar

The Kevlar Project.

## Dependencies

- [pre-commit](https://pre-commit.com/)

## Install

Please run all the steps here to get a fully functional development environment.

To clone the git repo:
```
git clone git@github.com:mikesklar/kevlar.git

To set up your conda environment (note that you may substitute `mamba` here for `conda` and the install will be substantially faster):
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
