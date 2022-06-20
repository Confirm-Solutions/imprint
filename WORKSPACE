load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

rules_python_version = "740825b7f74930c62f44af95c9a4c1bd428d2c53"  # Latest @ 2021-06-23

# Python rules
http_archive(
    name = "rules_python",
    sha256 = "09a3c4791c61b62c2cbc5b2cbea4ccc32487b38c7a2cc8f87a794d7a659cc742",
    strip_prefix = "rules_python-{}".format(rules_python_version),
    url = "https://github.com/bazelbuild/rules_python/archive/{}.zip".format(rules_python_version),
)

# GoogleTest/GoogleMock framework. Used by most unit-tests
http_archive(
    name = "com_google_googletest",
    sha256 = "205ddbea89a0dff059cd681f3ec9b0a6c12de7036a04cd57f0254105257593d9",
    strip_prefix = "googletest-13a433a94dd9c7e55907d7a9b75f44ff82f309eb",
    urls = ["https://github.com/google/googletest/archive/13a433a94dd9c7e55907d7a9b75f44ff82f309eb.zip"],
)

# Google benchmark
http_archive(
    name = "com_github_google_benchmark",
    sha256 = "59f918c8ccd4d74b6ac43484467b500f1d64b40cc1010daa055375b322a43ba3",
    strip_prefix = "benchmark-16703ff83c1ae6d53e5155df3bb3ab0bc96083be",
    urls = ["https://github.com/google/benchmark/archive/16703ff83c1ae6d53e5155df3bb3ab0bc96083be.zip"],
)

# Rules CC
http_archive(
    name = "rules_cc",
    sha256 = "9a446e9dd9c1bb180c86977a8dc1e9e659550ae732ae58bd2e8fd51e15b2c91d",
    strip_prefix = "rules_cc-262ebec3c2296296526740db4aefce68c80de7fa",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/262ebec3c2296296526740db4aefce68c80de7fa.zip"],
)

# PyBind11 Bazel
PYBIND_BAZEL_VERSION = "72cbbf1fbc830e487e3012862b7b720001b70672"

PYBIND_VERSION = "2.9.1"

http_archive(
    name = "pybind11_bazel",
    sha256 = "fec6281e4109115c5157ca720b8fe20c8f655f773172290b03f57353c11869c2",
    strip_prefix = "pybind11_bazel-{}".format(PYBIND_BAZEL_VERSION),
    urls = ["https://github.com/pybind/pybind11_bazel/archive/{}.zip".format(PYBIND_BAZEL_VERSION)],
)

# We still require the pybind library.
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    sha256 = "c6160321dc98e6e1184cc791fbeadd2907bb4a0ce0e447f2ea4ff8ab56550913",
    strip_prefix = "pybind11-{}".format(PYBIND_VERSION),
    urls = ["https://github.com/pybind/pybind11/archive/v{}.tar.gz".format(PYBIND_VERSION)],
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")

python_configure(name = "local_config_python")

# fmt
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

# ====================================
# GOOGLE TCMALLOC + DEPENDENCIES
# ====================================

http_archive(
    name = "rules_fuzzing",
    sha256 = "a5734cb42b1b69395c57e0bbd32ade394d5c3d6afbfe782b24816a96da24660d",
    strip_prefix = "rules_fuzzing-0.1.1",
    urls = ["https://github.com/bazelbuild/rules_fuzzing/archive/v0.1.1.zip"],
)

# Protobuf
load("@rules_fuzzing//fuzzing:repositories.bzl", "rules_fuzzing_dependencies")

rules_fuzzing_dependencies()

load("@rules_fuzzing//fuzzing:init.bzl", "rules_fuzzing_init")

rules_fuzzing_init()

http_archive(
    name = "com_google_absl",
    sha256 = "92d469a1a652fd1944398e560bd0d92ee8e3affbd61ed41fca89bb624b59109e",
    strip_prefix = "abseil-cpp-04bde89e5cb33bf4a714a5496fac715481fc48311",
    urls = ["https://github.com/abseil/abseil-cpp/archive/04bde89e5cb33bf4a714a5496fac715481fc48311.zip"],
)

http_archive(
    name = "com_google_tcmalloc",
    sha256 = "2e5e6755e02b0275b1333199c2a128a57c0d48ec8838fdca9baccf3b0e939ad6",
    strip_prefix = "tcmalloc-a3717bc4fcade63c642f9b991fbdd64299896762",
    urls = ["https://github.com/google/tcmalloc/archive/a3717bc4fcade63c642f9b991fbdd64299896762.zip"],
)

# ====================================
# EIGEN
# ====================================

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
    hdrs = glob(['Eigen/**', 'unsupported/Eigen/**']),
    visibility = ['//visibility:public'],
)
""",
    sha256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
    strip_prefix = "eigen-{}".format(EIGEN_VERSION),
    urls = ["https://gitlab.com/libeigen/eigen/-/archive/{0}/eigen-{0}.tar.gz".format(EIGEN_VERSION)],
)

_BOOST_COMMIT = "d8626c9d2d937abf6a38a844522714ad72e63281"

http_archive(
    name = "com_github_scipy_boost",
    build_file_content =
        """
cc_library(
    name = 'boost',
    srcs = [],
    includes = ['.'],
    hdrs = glob(['boost/**']),
    visibility = ['//visibility:public'],
)
""",
    sha256 = "496064bba545eb218179c0fa479304ac396ecca9f02ba6e0d3d4cc872f3569fa",
    strip_prefix = "boost-headers-only-%s" % _BOOST_COMMIT,
    urls = [
        "https://github.com/scipy/boost-headers-only/archive/%s.zip" % _BOOST_COMMIT,
    ],
)
