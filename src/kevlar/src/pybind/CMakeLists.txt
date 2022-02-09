pybind11_add_module(pykevlar
    # main export
    ${CMAKE_CURRENT_SOURCE_DIR}/pybind.cpp

    # util
    ${CMAKE_CURRENT_SOURCE_DIR}/util/range/grid_range.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/range/range.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/util.cpp

    # model
    ${CMAKE_CURRENT_SOURCE_DIR}/model/binomial_control_k_treatment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model/model.cpp
    )
target_include_directories(pykevlar PRIVATE ${KEVLAR_SOURCEDIR})
target_link_libraries(pykevlar PRIVATE
    ${PROJECT_NAME}
    Eigen3::Eigen)

install(TARGETS pykevlar
  COMPONENT python
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
