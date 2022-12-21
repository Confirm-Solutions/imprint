# Imprint

Imprint is a library to validate clinical trial designs.

[![Gitter](https://img.shields.io/gitter/room/confirm_imprint/community)](https://gitter.im/confirm_imprint/community#)
![example workflow](https://github.com/Confirm-Solutions/imprint/actions/workflows/test.yml/badge.svg)

## Installing Imprint for development.

(Soon, we will have a separate pathway for users to install via PyPI/pip)

1. If you do not have conda installed already, please install it. There are
   many ways to get conda. We recommend installing `Mambaforge` which is a
   conda installation wwith `mamba` installed by default and set to use
   `conda-forge` as the default set of package repositories. [CLICK HERE for
   installers and installation
   instructions.](https://github.com/conda-forge/miniforge#mambaforge)
2. Clone the git repo:

   ```bash
   git clone git@github.com:Confirm-Solutions/imprint.git
   ```

3. Set up your imprint conda environment. The list of packages that will be
   installed inside your conda environment can be seen
   in [`pyproject.toml`](pyproject.toml).

   ```bash
   mamba update -y conda
   # create a development virtual environment with useful tools
   mamba env create
   conda activate confirm
   # install the confirm package plus development tools
   poetry install --with=dev,test,cloud,cloud_dev
   ```
   
## Committing code

In order to commit code and pass the pre-commit checks, you will need to install `go` and `gitleaks` with `brew install go gitleaks`.

## Getting started understanding imprint

[Tutorial: z-test](./tutorials/basket.ipynb)
[Tutorial: Fisher exact](./tutorials/basket.ipynb)
[Tutorial: Three arm Bayesian Basket trial](./tutorials/basket.ipynb)

## References

- [Adaptive Experiments and a Rigorous Framework for Type I Error Verification and Computational Experiment Design](https://arxiv.org/abs/2205.09369)
