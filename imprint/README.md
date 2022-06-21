# Imprint C++ Core Engine

## Overview

The Imprint C++ core engine, `imprint`, implements the core components of the
system.

## Dependencies

If you have set up a conda environment following the instructions in the main repo README, you should already have a C++ toolchain installed along with Bazel. If not, we require a C++ toolchain that supports C++-17 and [OpenMP](https://www.openmp.org/) and an installation of [Bazel](https://bazel.build/)

Suggested compilers:
- [GCC >= 9.3.0](https://gcc.gnu.org/)
- [Clang >= 11.0.0](https://clang.llvm.org/)

## Build

__Note: `CMake` build has been deprecated and is not maintained.__

__Note: On Linux, it's best to specify whether you want to use `clang` or `gcc`.
Add the appropriate flag to each `bazel` call below:
```
# For gcc
# For clang
bazel ... --config=gcc
bazel ... --config=clang
```

__To build `imprint`__:
```
bazel build //imprint:imprint 
```
Note that `imprint` is a header-only library,
so this will simply collect all the headers and register its dependencies.
For release mode, add the flag `-c opt` after `build`.
For debug mode, add the flag `-c dbg` after `build`.

__To run all tests__:
```
bazel test -c dbg //imprint/test/... 
```

__To run a particular test__:
```
bazel test -c dbg //imprint/test:name-of-test
```
where `name-of-test` is the same name as the subdirectory in `test/`
besides `testutil`.

__To run the benchmarks__:
```
bazel run -c opt //imprint/benchmark:name-of-benchmark
```
where `name-of-benchmark` is the same name of 
the benchmark `.cpp` file in the `benchmark/` folder.

## Supported Models

| Model | Description |
| ----- | ----------- |
| Binomial Control + k Treatment | Phase II/III trial with `k+1` total number of arms (1 control, k treatments) and each patient modeled as a `Bernoulli(p_i)` where `p_i` is the null probability of response for arm `i`. Phase II makes a selection of the treatment arm with most number of responses and Phase III constructs the [unpaired z-test](https://en.wikipedia.org/wiki/Paired_difference_test#Power_of_the_paired_Z-test) between the selected and control arms. |
| Exponential Control + k Treatment | Phase III trial with `k+1` total number of arms (1 control, k treatments) and each patient modeled as an `Exponential(lambda_i)` where `lambda_i` is the hazard for arm `i`. Currently, it only supports 1 treatment arm. Phase III makes a selection of the treatment arm with most number of responses and constructs the [logrank test](https://en.wikipedia.org/wiki/Logrank_test) between the treatment and control arms. |
