#!/bin/zsh
eval "$(conda shell.zsh hook)"
conda activate imprint
bazel build //python:pyimprint/core.so
ln -sf ./bazel-bin/python/pyimprint/core.so python/pyimprint/core.so