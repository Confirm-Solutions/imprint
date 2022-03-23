cc_library(
    name = "kevlar",
    hdrs = glob(["include/**"]),
    copts = ["-fopenmp"],
    defines = [
        # force Eigen to not use openmp to parallelize
        # we will use openmp to divide up simulation jobs
        # so there is no room for Eigen to further parallelize
        "EIGEN_DONT_PARALLELIZE",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
        "@kthohr_stats",
    ],
)
