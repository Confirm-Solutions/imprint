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
    EXAMPLES
    binomial_control_2_treatment_tune
    binomial_control_2_treatment_fit
    exp_control_treatment_tune
)

set(
    EXAMPLE_LIBS
    Eigen3::Eigen
)

if (KEVLAR_HAS_PTHREAD)
    list(APPEND EXAMPLE_LIBS Threads::Threads)
endif()

foreach(example ${EXAMPLES})
    add_executable(${example}
        ${CMAKE_CURRENT_SOURCE_DIR}/${example}.cpp)
    target_link_libraries(${example}
        ${PROJECT_NAME}
        ${EXAMPLE_LIBS})
    if (KEVLAR_HAS_PTHREAD)
        target_compile_definitions(${example} PRIVATE
            ${KEVLAR_HAS_PTHREAD_MACRO})
    endif()
endforeach()
