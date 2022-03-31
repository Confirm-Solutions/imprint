cc_library(
    name = "testutil",
    srcs = glob([
        "testutil/**/*.hpp",
        "testutil/**/*.cpp",
    ]),
    hdrs = glob([
        "testutil/**/*.hpp",
    ]),
    includes = ["."],
    deps = [
        "//kevlar",
    ],
)

[cc_test(
    name = type_,
    srcs = glob(["{}/**/*.cpp".format(type_)]),
    defines = ["EIGEN_INITIALIZE_MATRICES_BY_NAN"],
    deps = [
        ":testutil",
        "//kevlar",
        "@com_google_googletest//:gtest_main",
    ],
) for type_ in [
    "grid",
    "model",
    "stats",
    "util",
]]
