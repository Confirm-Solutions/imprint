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
        "//src/kevlar",
    ],
)

[cc_test(
    name = type_,
    srcs = glob(["{}/**/*.cpp".format(type_)]),
    deps = [
        ":testutil",
        "//src/kevlar",
        "@com_google_googletest//:gtest_main",
    ],
) for type_ in [
    "grid",
    "model",
    "stats",
    "util",
]]
