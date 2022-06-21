# Set compile flags, libs, include for building tests
set(IMPRINT_TEST_CXXFLAGS -g -Wall)
set(IMPRINT_TEST_LIBS
    ${PROJECT_NAME}
    GTest::gtest_main
    Eigen3::Eigen)
set(IMPRINT_TEST_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(IMPRINT_TEST_CXXFLAGS
        ${IMPRINT_TEST_CXXFLAGS}
        -Werror -Wextra -Wpedantic)
endif()
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(IMPRINT_TEST_CXXFLAGS ${IMPRINT_TEST_CXXFLAGS} -fopenmp)
    set(IMPRINT_TEST_LIBS ${IMPRINT_TEST_LIBS} -fopenmp)
endif()
if (IMPRINT_ENABLE_COVERAGE)
    set(IMPRINT_TEST_LIBS ${IMPRINT_TEST_LIBS} gcov)
endif()

function(imprint_add_test name)
    target_compile_options(${name} PRIVATE ${IMPRINT_TEST_CXXFLAGS})
    target_include_directories(${name} PRIVATE ${IMPRINT_TEST_INCLUDES})
    target_link_libraries(${name} ${IMPRINT_TEST_LIBS})
    add_test(${name} ${name})
endfunction()

########################################################################
# Utility TEST
########################################################################

add_executable(utility_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/util/algorithm_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/d_ary_int_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/math_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/progress_bar_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/types_unittest.cpp
    )
imprint_add_test(utility_unittest)

########################################################################
# Grid TEST
########################################################################

add_executable(grid_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/grid/gridder_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/grid/grid_range_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/grid/tile_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/grid/hyperplane_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/grid/utils_unittest.cpp
    )
imprint_add_test(grid_unittest)

########################################################################
# Model TEST
########################################################################

add_executable(model_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/model/binomial_control_k_treatment_unittest.cpp
    )
imprint_add_test(model_unittest)

########################################################################
# Stats TEST
########################################################################

add_executable(stats_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/stats/inter_sum_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stats/upper_bound_unittest.cpp
    )
imprint_add_test(stats_unittest)
