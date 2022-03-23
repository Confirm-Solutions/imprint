load("//:configs.bzl", "define_compiler_config_settings")

define_compiler_config_settings()

COMPILER_SPECIFIC_COPTS = select({
    ":llvm_compiler" : [
        "-fopenmp",
    ],
    ":gcc_compiler" : [
        "-fopenmp",
    ],
})

COMPILER_SPECIFIC_LINKOPTS = select({
    ":llvm_compiler" : [
        "-lomp",
    ],
    ":gcc_compiler" : [
        "-fopenmp",
    ],
})

cc_library(
    name = "kevlar",
    includes = ["include"],
    hdrs = glob(["include/**"]),
    copts = COMPILER_SPECIFIC_COPTS,
    defines = [
        # force Eigen to not use openmp to parallelize
        # we will use openmp to divide up simulation jobs
        # so there is no room for Eigen to further parallelize
        "EIGEN_DONT_PARALLELIZE",
    ],
    linkopts = COMPILER_SPECIFIC_LINKOPTS,
    visibility = ["//visibility:public"],
    deps = ["@eigen", "@kthohr_stats"],
)
