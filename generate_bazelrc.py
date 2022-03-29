import os
import re
import subprocess
from sys import platform

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(ROOT_DIR, '.bazelrc')


def run_cmd(cmd):
    process = subprocess.Popen(
        cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if stderr.decode('utf-8') != "":
        raise RuntimeError(stderr)
    return stdout.decode('utf-8').rstrip()


def main():
    with open(OUT_PATH, 'w') as f:
        f.write(
'''
build --cxxopt="-std=c++17"
build --cxxopt="-Wall"

# Linux GCC
build:gcc --action_env=CC=gcc
build:gcc --linkopt -fopenmp

# Linux Clang
build:clang --action_env=CC=clang
build:clang --linkopt -fopenmp
'''
        )

        # Only run these on MacOS
        if platform == 'darwin':
            # get canonical brew path
            clang_prefix = run_cmd('brew --prefix llvm')
            clang_path = os.path.join(clang_prefix, 'bin/clang')

            # get canonical omp path
            omp_prefix = run_cmd('brew --prefix libomp')
            omp_path = os.path.join(omp_prefix, 'lib')

            f.write(
f'''
# Mac Homebrew Clang
build:mac --action_env=CC={clang_path}
build:mac --linkopt -L{omp_path}
build:mac --linkopt -lomp
# Tell Bazel not to use the full Xcode toolchain on Mac OS
build:mac --repo_env=BAZEL_USE_CPP_ONLY_TOOLCHAIN=1
'''
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


if __name__ == '__main__':
    main()
