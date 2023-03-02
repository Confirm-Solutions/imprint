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
   mamba update -y conda mamba
   # create a development virtual environment with useful tools
   mamba env create
   conda activate imprint
   # the following line makes poetry install into already activated
   # environment.
   poetry config virtualenvs.create false --local
   # install the imprint package plus development tools
   poetry install --with=dev,test

   # OPTIONALLY: Install a CUDA-enabled JAX build on Linux
   pip install --upgrade -r requirements-jax-cuda.txt
   ```
   
## Getting started understanding imprint

- [Tutorial: Z Test](./tutorials/ztest.ipynb)
- [Tutorial: Chi-Squared Test](./tutorials/chisq_test.ipynb)
- [Tutorial: T Test](./tutorials/t_test.ipynb)
- [Tutorial: T Test Adaptive](./tutorials/t_test_adaptive.ipynb)
- [Tutorial: Fisher's Exact Test](./tutorials/fisher_exact.ipynb)
- [Tutorial: Three-arm Bayesian Basket Trial](./tutorials/basket.ipynb)

## References

- The main paper describing what imprint does: [A Rigorous Framework for Automated Design Assessment and Type I Error Control: Methods and Demonstrative Examples](https://arxiv.org/abs/2212.10042)
- Some older discussion of similar but obsolete methods can be found in Chapter 5 here: [Adaptive Experiments and a Rigorous Framework for Type I Error Verification and Computational Experiment Design](https://arxiv.org/abs/2205.09369)
