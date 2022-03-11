cc_binary(
    name = "benchmark",
    srcs = ["direct_bayes.cpp"],
    malloc = "@com_google_tcmalloc//tcmalloc",
    deps = [
        "//src/kevlar",
        "@com_github_google_benchmark//:benchmark_main",
    ],
)
