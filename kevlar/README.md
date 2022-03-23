# Kevlar C++ Core Engine

## Overview

The Kevlar C++ core engine, `kevlar`,
implements the core components of the system.

## Dependencies

In addition to a C++ compiler that can support C++-17 and [OpenMP](https://www.openmp.org/):
- [Bazel](https://bazel.build/)

Suggested compilers:
- [GCC >= 9.3.0](https://gcc.gnu.org/)
- [Clang >= 11.0.0](https://clang.llvm.org/)

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

## Build

__Note: `CMake` build has been deprecated and is not maintained.__

__Note: If the system-provided `clang` complains about unrecognized option `-fopenmp`
in the build procedures below,
the user is expected to use the Homebrew-provided `clang`
and define the environment variables `CC, CXX` before calling `bazel`:__
```
brew install llvm
CC=/path/to/brew/clang CXX=/path/to/brew/clang++ bazel ...
```

__Note: For Linux users who wish to use `clang`:__
```
sudo apt install clang-11 --install-suggests
```

__To build `kevlar`__:
```
bazel build //kevlar:kevlar 
```
Note that `kevlar` is a header-only library,
so this will simply collect all the headers and register its dependencies.
For release mode, add the flag `-c opt` after `build`.
For debug mode, add the flag `-c dbg` after `build`.

__To run all tests__:
```
bazel test -c dbg //kevlar/test/... 
```

__To run a particular test__:
```
bazel test -c dbg //kevlar/test:name-of-test
```
where `name-of-test` is the same name as the subdirectory in `test/`
besides `testutil`.

__To run the benchmarks__:
```
bazel run -c opt //kevlar/benchmark:name-of-benchmark
```
where `name-of-benchmark` is the same name of 
the benchmark `.cpp` file in the `benchmark/` folder.

## Supported Models

| Model | Description |
| ----- | ----------- |
| Binomial Control + k Treatment | Phase II/III trial with `k+1` total number of arms (1 control, k treatments) and each patient modeled as a `Bernoulli(p_i)` where `p_i` is the null probability of response for arm `i`. Phase II makes a selection of the treatment arm with most number of responses and Phase III constructs the [unpaired z-test](https://en.wikipedia.org/wiki/Paired_difference_test#Power_of_the_paired_Z-test) between the selected and control arms. |
| Exponential Control + k Treatment | Phase III trial with `k+1` total number of arms (1 control, k treatments) and each patient modeled as an `Exponential(lambda_i)` where `lambda_i` is the hazard for arm `i`. Currently, it only supports 1 treatment arm. Phase III makes a selection of the treatment arm with most number of responses and constructs the [logrank test](https://en.wikipedia.org/wiki/Logrank_test) between the treatment and control arms. |
