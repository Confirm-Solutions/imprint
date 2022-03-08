load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "com_google_absl",
    sha256 = "aabf6c57e3834f8dc3873a927f37eaf69975d4b28117fc7427dfb1c661542a87",
    strip_prefix = "abseil-cpp-98eb410c93ad059f9bba1bf43f5bb916fc92a5ea",
    urls = ["https://github.com/abseil/abseil-cpp/archive/98eb410c93ad059f9bba1bf43f5bb916fc92a5ea.zip"],
)

http_archive(
    name = "rules_cc",
    sha256 = "9a446e9dd9c1bb180c86977a8dc1e9e659550ae732ae58bd2e8fd51e15b2c91d",
    strip_prefix = "rules_cc-262ebec3c2296296526740db4aefce68c80de7fa",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/262ebec3c2296296526740db4aefce68c80de7fa.zip"],
)

http_archive(
    name = "com_google_googletest",
    sha256 = "5cf189eb6847b4f8fc603a3ffff3b0771c08eec7dd4bd961bfd45477dd13eb73",
    strip_prefix = "googletest-609281088cfefc76f9d0ce82e1ff6c30cc3591e5",
    urls = ["https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip"],
)

http_archive(
    name = "fmtlib",
    patch_cmds = [
        "mv support/bazel/.bazelrc .bazelrc",
        "mv support/bazel/.bazelversion .bazelversion",
        "mv support/bazel/BUILD.bazel BUILD.bazel",
        "mv support/bazel/WORKSPACE.bazel WORKSPACE.bazel",
    ],
    sha256 = "23778bad8edba12d76e4075da06db591f3b0e3c6c04928ced4a7282ca3400e5d",
    strip_prefix = "fmt-8.1.1",
    urls = ["https://github.com/fmtlib/fmt/releases/download/8.1.1/fmt-8.1.1.zip"],
)

http_archive(
    name = "cppitertools",
    sha256 = "204fa0c872c0f34c9263862a5f4b93a9858fde9fdae8064febc7683863f1497d",
    strip_prefix = "cppitertools-2.1",
    urls = ["https://github.com/ryanhaining/cppitertools/archive/refs/tags/v2.1.zip"],
)

EIGEN_VERSION = "3.4.0"

http_archive(
    name = "eigen",
    build_file_content =
        """
# TODO(keir): Replace this with a better version, like from TensorFlow.
# See https://github.com/ceres-solver/ceres-solver/issues/337.
cc_library(
    name = 'eigen',
    srcs = [],
    includes = ['.'],
    hdrs = glob(['Eigen/**']),
    visibility = ['//visibility:public'],
)
""",
    sha256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
    strip_prefix = "eigen-{}".format(EIGEN_VERSION),
    urls = ["https://gitlab.com/libeigen/eigen/-/archive/{0}/eigen-{0}.tar.gz".format(EIGEN_VERSION)],
)
