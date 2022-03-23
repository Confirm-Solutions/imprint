# https://github.com/bazelbuild/bazel/issues/12707
# Why is there non-existent documentation on setting compiler-specific flags??
# This macros defines compiler-specific config settings.
# All compiler-specific flags/options should be selected based on these settings.
def define_compiler_config_settings():
    native.config_setting(
        name = "llvm_compiler",
        flag_values = {"@bazel_tools//tools/cpp:compiler": "clang"},
    )

    native.config_setting(
        name = "gcc_compiler",
        flag_values = {"@bazel_tools//tools/cpp:compiler": "gcc"},
    )
