# All macro tests
set(
    BENCHMARKS
    bench_binomial_2_arm
    binomial_control_k_treatment_tune
)

foreach( benchmark ${BENCHMARKS} )
    add_executable(${benchmark} ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cpp)
    target_compile_options(${benchmark} PRIVATE -std=c++17)
    target_link_libraries(${benchmark}
        ${PROJECT_NAME}
        benchmark::benchmark_main
        Eigen3::Eigen)
endforeach()
