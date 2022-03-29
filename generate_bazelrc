#!/usr/bin/env python3

import os
import subprocess
from sys import platform
from string import Template

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(ROOT_DIR, '.bazelrc')


def run_cmd(cmd):
    try:
        output = subprocess.check_output(
            cmd.split(' '), stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
        raise RuntimeError(output)
    return output.rstrip()


def main():
    with open(OUT_PATH, 'w') as f:
        f.write(
'''
build --cxxopt="-std=c++17"
build --cxxopt="-Wall"
build --cxxopt="-fopenmp"

# Linux GCC
build:gcc --action_env=CC=gcc
build:gcc --action_env=CXX=g++
build:gcc --linkopt -fopenmp

# Linux Clang
build:clang --action_env=CC=clang
build:clang --action_env=CXX=clang++
build:clang --linkopt -fopenmp
'''
        )

        # Only run these on MacOS
        if platform == 'darwin':
            # get canonical brew path
            clang_prefix = run_cmd('brew --prefix llvm')
            clang_path = os.path.join(clang_prefix, 'bin')

            # get canonical omp path
            omp_prefix = run_cmd('brew --prefix libomp')
            omp_path = os.path.join(omp_prefix, 'lib')

            mac_build = Template(
'''
# Mac Homebrew Clang
build:mac --action_env=CC=${clang_path}/clang
build:mac --action_env=CXX=${clang_path}/clang++
build:mac --linkopt -L${omp_path}
build:mac --linkopt -lomp
# Tell Bazel not to use the full Xcode toolchain on Mac OS
build:mac --repo_env=BAZEL_USE_CPP_ONLY_TOOLCHAIN=1
'''
            )
            f.write(
                mac_build.substitute(
                    clang_path=clang_path,
                    omp_path=omp_path,
                )
            )

        # Add asan build
        f.write(
'''
build:asan --strip=never
build:asan --copt -fsanitize=address
build:asan --copt -DADDRESS_SANITIZER
build:asan --copt -O1
build:asan --copt -g
build:asan --copt -fno-omit-frame-pointer
build:asan --linkopt -fsanitize=address
'''
        )

        # TODO: add ubsan + msan builds also


if __name__ == '__main__':
    main()
