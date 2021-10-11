add_executable(binomial_2_arm
    ${CMAKE_CURRENT_SOURCE_DIR}/binomial_2_arm.cpp)
target_link_libraries(binomial_2_arm
    ${PROJECT_NAME}
    Eigen3::Eigen)
