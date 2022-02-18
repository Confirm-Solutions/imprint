########################################################################
# Utility TEST
########################################################################

add_executable(utility_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/util/algorithm_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/d_ary_int_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/grid_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/math_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/progress_bar_unittest.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/util/range/grid_range_unittest.cpp
    )

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(utility_unittest PRIVATE -Werror -Wextra -Wpedantic)
endif()
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(utility_unittest PRIVATE -fopenmp)
    target_link_libraries(utility_unittest -fopenmp)
endif()
target_compile_options(utility_unittest PRIVATE -g -Wall)
target_include_directories(utility_unittest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
if (KEVLAR_ENABLE_COVERAGE)
    target_link_libraries(utility_unittest gcov)
endif()
target_link_libraries(utility_unittest
    ${PROJECT_NAME}
    GTest::gtest_main
    Eigen3::Eigen)
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(utility_unittest pthread)
endif()
add_test(utility_unittest utility_unittest)

########################################################################
# Model TEST
########################################################################

add_executable(model_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/model/binomial_control_k_treatment_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/intersum_unittest.cpp
    )

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(model_unittest PRIVATE -Werror -Wextra -Wpedantic)
endif()
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(model_unittest PRIVATE -fopenmp)
    target_link_libraries(model_unittest -fopenmp)
endif()
target_compile_options(model_unittest PRIVATE -g -Wall)
target_compile_definitions(model_unittest PRIVATE
    ${KEVLAR_HAS_PTHREAD_MACRO})
target_include_directories(model_unittest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${KEVLAR_SOURCEDIR})
if (KEVLAR_ENABLE_COVERAGE)
    target_link_libraries(model_unittest gcov)
endif()
target_link_libraries(model_unittest
    ${PROJECT_NAME}
    GTest::gtest_main
    Eigen3::Eigen)
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(model_unittest pthread)
endif()
add_test(model_unittest model_unittest)

