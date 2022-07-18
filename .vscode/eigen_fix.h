/*
 * Proposed fix with IntelliSense that keeps raising error with Eigen classes
 * about incomplete types.
 * https://github.com/microsoft/vscode-cpptools/issues/7413#issuecomment-1105063602
 * This file is force-included in c_cpp_properties.json.
 */
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif