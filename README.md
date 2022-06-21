# Imprint

Imprint is a library to validate clinical trial designs.

![example workflow](https://github.com/Confirm-Solutions/imprint/actions/workflows/test.yml/badge.svg)

## Dependencies

The most important dependencies are:

- [conda](https://docs.conda.io/projects/conda/en/latest/index.html)
  - [Anaconda](https://www.anaconda.com/)
  - [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Python >= 3.9](https://www.python.org/)

## Installing Imprint for development

NOTE: In the future, we will produce PyPI and conda-forge packages to ease the installation process for users. This will reduce the installation process to one or two steps. The current process is oriented at a developer of imprint.

Please run all the steps here to get a fully functional development environment.

1. If you do not have conda installed already, please install it. There are
   many ways to get conda. We recommend installing `Mambaforge` which is a
   conda installation wwith `mamba` installed by default and set to use
   `conda-forge` as the default set of package repositories. [CLICK HERE for
   installers and installation
   instructions.](https://github.com/conda-forge/miniforge#mambaforge)
2. Install Bazel. On Mac, you can just run `brew install bazelisk`. On Ubuntu
   Linux, please follow the [instructions
   here](https://docs.bazel.build/versions/main/install-ubuntu.html).
3. Clone the git repo:

    ```bash
    git clone git@github.com:Confirm-Solutions/imprint.git
    ```

4. Set up your imprint conda environment (note that you may substitute `mamba`
   here for `conda` and the install will be substantially faster). The list of
   packages that will be installed inside your conda environment can be seen in
   the [`environment.yml` file](../environment.yml).

    ```bash
    cd imprint/
    conda update -y conda
    conda env create
    conda activate imprint
    ```

5. To set up pre-commit for this git repo:

    ```bash
    pre-commit install
    ```

6. To set up your bazel configuration for building C++. **See below to install bazel.**

    ```bash
    ./generate_bazelrc
    ```

7. Build and install the `pyimprint` package.

    ```bash
    bazel build //python:pyimprint_wheel
    pip install bazel-bin/python/dist/pyimprint-0.1-py3-none-any.whl
    ```

8. (it's okay to skip this step if this is your first time installing imprint) To recompile and reinstall the pyimprint package after making changes to the C++ backend, run the following:

    ```bash
    bazel build //python:pyimprint_wheel
    pip install --force-reinstall bazel-bin/python/dist/pyimprint-0.1-py3-none-any.whl
    ```

9. Finally, check that the installation process was successful by running one of our example scripts:

    ```bash
    bazel run -c opt //python/example:simple_selection -- main
    ```

## Getting started understanding imprint

[Please check out the tutorial where we analyze a three arm basket trial here.](./research/berry/tutorial.ipynb)

## Developing the Imprint C++ core engine

Most users will not need to work directly with the core C++, instead working entirely through the Python interface.

[Instructions for developing the C++ core engine are available in the subfolder](./imprint/README.md)
