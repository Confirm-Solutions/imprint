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
    "binomial_control_k_treatment_fit",
    "direct_bayes",
]]
