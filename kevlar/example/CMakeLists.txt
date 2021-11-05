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

add_executable(binomial_control_2_treatment
    ${CMAKE_CURRENT_SOURCE_DIR}/binomial_control_2_treatment.cpp)
target_link_libraries(binomial_control_2_treatment
    ${PROJECT_NAME}
    Threads::Threads
    Eigen3::Eigen)
if (KEVLAR_HAS_PTHREAD)
    target_compile_definitions(binomial_control_2_treatment PRIVATE
        ${KEVLAR_HAS_PTHREAD_MACRO})
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_link_libraries(binomial_control_2_treatment
            pthread)
    endif()
endif()
