[cc_binary(
    name = type_,
    srcs = ["{}.cpp".format(type_)],
    deps = [
        "//src/kevlar",
        "@com_github_google_benchmark//:benchmark_main",
    ],
) for type_ in [
    "direct_bayes",
    "bench_binomial_2_arm",
    "binomial_control_k_treatment_fit",
]]
