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
conda update -y conda
conda env create
conda activate kevlar
```

To set up pre-commit:
```
pre-commit install
```

To set up your bazel configuration for building C++. **See below to install bazel.**
```
./generate_bazelrc
```

From here, we refer to the installation instructions
for each of the sub-components:

- [pykevlar](./python/README.md): Kevlar Python package.
- [kevlar](./kevlar/README.md): Kevlar C++ core engine.


## Install Bazel in the typical way for your OS:

### Mac OS

To install the dependencies:
```
brew install bazelisk 
```

### Ubuntu Linux

To install the dependencies:
```
mkdir -p /some/dir
curl -Lo /some/dir/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.1.0/bazelisk-linux-amd64
chmod +x /some/dir/bazel
```
where `/some/dir` is a directory to store the `bazel` binary.
Note that for a system-wide install, the user may need to call under `sudo`.
For a local install, the user should add `/some/dir` to `PATH`:
```
export PATH="/some/dir:$PATH"
```