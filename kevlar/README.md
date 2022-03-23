# Kevlar C++ Core Engine

## Overview

This is the C++ core engine for Kevlar.
The goal of this engine is to serve as a testbed of highly optimized 
and configurable simulations for popular models used in clinical trials.

## Dependencies

Kevlar requires a recent version of Bazel and a C++17-capable C++ compiler:

### Mac OS

```
brew install bazelisk llvm
```

### Ubuntu Linux

```
sudo curl -Lo /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.1.0/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazel
sudo apt install clang-11 --install-suggests
```

## Build

__Note that we deprecated `CMake` build and is not maintained.__

To build `kevlar`, run the following:
```
bazel build //kevlar:kevlar
```
Note that `kevlar` is a header-only library,
so this will simply collect all the headers.

To run all tests, run the following:
```
bazel test -c dbg //kevlar/test/... 
```
If there are compilation issues related to `C++17` standard not being set,
pass an additional flag `--cxxopt='-std=c++17'`.

To run a particular test, run the following:
```
bazel test -c dbg //kevlar/test:name-of-test
```
where `name-of-test` is the same name as the subdirectory in `kevlar/test/`
besides `testutil`.

To run the examples in `example/`, do the following:
```
bazel run //kevlar/example:name-of-example
```
where `name-of-example` is the same name of the example `.cpp` file in `example/` folder.

## Supported Models

| Model | Example | Description |
| ----- | ------- | ----------- |
| Binomial Control + k Treatment | [binomial_control_2_treatment_tune.cpp](example/binomial_control_2_treatment_tune.cpp) [binomial_control_2_treatment_fit.cpp](example/binomial_control_2_treatment_fit.cpp) | Clinical trial Phase II/III with `k+1` total number of arms (1 control, k treatments) and each patient modeled as a `Bernoulli(p_i)` where `p_i` is the null probability of response for arm `i`. Phase II makes a selection of the treatment arm with most number of responses and constructs the [unpaired z-test](https://en.wikipedia.org/wiki/Paired_difference_test#Power_of_the_paired_Z-test) against the control arm. Currently only supports rectangular null-space grid. |
