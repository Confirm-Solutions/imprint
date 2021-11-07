#TODO: disable temporarily
#add_executable(binomial_2_arm
#    ${CMAKE_CURRENT_SOURCE_DIR}/binomial_2_arm.cpp)
#target_link_libraries(binomial_2_arm
#    ${PROJECT_NAME}
#    Threads::Threads
#    Eigen3::Eigen)
#if (KEVLAR_HAS_PTHREAD)
#    target_compile_definitions(binomial_2_arm PRIVATE
#        ${KEVLAR_HAS_PTHREAD_MACRO})
#    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
#        target_link_libraries(binomial_2_arm
#            pthread)
#    endif()
#endif()

set(
    BENCHMARKS
    binomial_control_2_treatment_tune
    binomial_control_2_treatment_fit
)

foreach(benchmark ${BENCHMARKS})
    add_executable(${benchmark}
        ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cpp)
    target_link_libraries(${benchmark}
        ${PROJECT_NAME}
        Threads::Threads
        Eigen3::Eigen)
    if (KEVLAR_HAS_PTHREAD)
        target_compile_definitions(${benchmark} PRIVATE
            ${KEVLAR_HAS_PTHREAD_MACRO})
    endif()
endforeach()
