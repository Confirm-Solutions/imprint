[cc_binary(
    name = type_,
    srcs = ["{}.cpp".format(type_)],
    # See GH-64
    # malloc = "@com_google_tcmalloc//tcmalloc",
    deps = [
        "//kevlar",
        "@com_github_google_benchmark//:benchmark_main",
    ],
) for type_ in [
    "bench_binomial_2_arm",
    "direct_bayes",
    "simple_selection_accum",
]]
