# Kevlar C++ Core Engine

## Overview

This is the C++ core engine for Kevlar.
The goal of this engine is to serve as a testbed of highly optimized 
and configurable simulations for popular models used in clinical trials.

## Dependencies

- [Eigen >= 3.3](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [CMake >= 3.7](https://cmake.org/)
- [GoogleTest](https://github.com/google/googletest) (_dev only_)

## Build

First, install Eigen. Download the latest release and follow the directions in the `INSTALL` file.
 
Simply run `./clean-build.sh [debug/release]` (choose one of `debug` or `release`).
This will create a `build` directory containing either `debug` or `release` sub-directory.
For testing purposes, it is recommended to build with `debug`.
For all other purposes, it is recommended to build with `release`.

One can also pass CMake options, e.g. `./clean-build.sh [debug/release] [CMake options...]`.
* Running the tests: if one wishes to run the tests and GoogleTest is installed
  locally in `/path/to/googletest`, then you must pass
  `-DGTest_DIR=/path/to/googletest/install/lib/cmake/GTest` where
  `/path/to/googletest/install` is the installation path to GoogleTest.
* If one wishes to build without tests, pass `-DKEVLAR_ENABLE_TEST=OFF`.
* To build without PThreads (probably because you're on a Mac) add `-DKEVLAR_HAS_PTHREAD=OFF`

To run tests, do the following:
```
cd build/[debug/release]
ctest
```

To run the examples in `example/`, do the following:
```
cd build/release
./name-of-example
```
where `name-of-example` is the same name of the example `.cpp` file in `example/` folder.

## Supported Models

| Model | Example | Description |
| ----- | ------- | ----------- |
| Binomial Control + k Treatment | [binomial_control_2_treatment_tune.cpp](example/binomial_control_2_treatment_tune.cpp) [binomial_control_2_treatment_fit.cpp](example/binomial_control_2_treatment_fit.cpp) | Clinical trial Phase II/III with `k+1` total number of arms (1 control, k treatments) and each patient modeled as a `Bernoulli(p_i)` where `p_i` is the null probability of response for arm `i`. Phase II makes a selection of the treatment arm with most number of responses and constructs the [unpaired z-test](https://en.wikipedia.org/wiki/Paired_difference_test#Power_of_the_paired_Z-test) against the control arm. Currently only supports rectangular null-space grid. |
