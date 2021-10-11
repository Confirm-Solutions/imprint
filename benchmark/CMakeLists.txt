add_executable(bench_binomial_2_arm
    ${CMAKE_CURRENT_SOURCE_DIR}/bench_binomial_2_arm.cpp)
target_link_libraries(bench_binomial_2_arm
    ${PROJECT_NAME}
    benchmark::benchmark_main
    Eigen3::Eigen)
