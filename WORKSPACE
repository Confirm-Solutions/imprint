load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# GoogleTest/GoogleMock framework. Used by most unit-tests.
http_archive(
    name = "com_google_googletest",
    sha256 = "205ddbea89a0dff059cd681f3ec9b0a6c12de7036a04cd57f0254105257593d9",
    strip_prefix = "googletest-13a433a94dd9c7e55907d7a9b75f44ff82f309eb",
    urls = ["https://github.com/google/googletest/archive/13a433a94dd9c7e55907d7a9b75f44ff82f309eb.zip"],
)

# Google benchmark.
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

# ====================================
# KTHOHR STATS + DEPENDENCIES
# ====================================

KTHOHR_GCEM_VERSION = "1.14.1"

KTHOHR_STATS_VERSION = "3.1.2"

http_archive(
    name = "kthohr_gcem",
    build_file_content =
        """
cc_library(
    name = "kthohr_gcem",
    includes = ['include'],
    hdrs = glob(['include/**']),
    visibility = ['//visibility:public'],
)
""",
    sha256 = "fd0860e89f47eeddf5a2280dd6fb3f9b021ce36fe8798116b3f703fa0e01409d",
    strip_prefix = "gcem-{}".format(KTHOHR_GCEM_VERSION),
    urls = ["https://github.com/kthohr/gcem/archive/refs/tags/v{0}.tar.gz".format(KTHOHR_GCEM_VERSION)],
)

http_archive(
    name = "kthohr_stats",
    build_file_content =
        """
cc_library(
    name = "kthohr_stats",
    includes = ['include'],
    hdrs = glob(['include/**']),
    visibility = ['//visibility:public'],
    deps = ['@kthohr_gcem'],
)
""",
    sha256 = "fe82c679dbed0cbea284ce077e2c2503afaec745658a3791f9fe5010e438305e",
    strip_prefix = "stats-{}".format(KTHOHR_STATS_VERSION),
    urls = ["https://github.com/kthohr/stats/archive/refs/tags/v{0}.tar.gz".format(KTHOHR_STATS_VERSION)],
)
