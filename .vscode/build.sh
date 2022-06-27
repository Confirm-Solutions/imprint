#!/bin/zsh
eval "$(conda shell.zsh hook)"
conda activate kevlar
rm -f bazel-bin/python/dist/*.whl
bazel build -c dbg //python:pykevlar_wheel
pip install --force-reinstall bazel-bin/python/dist/*.whl