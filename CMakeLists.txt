project(DD_ILP)

cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# C++14
add_compile_options(-std=c++14)

# compiler options
add_definitions(-march=native)

add_library(DD_ILP INTERFACE)
target_include_directories(DD_ILP INTERFACE . include )

include(ExternalProject)

list(APPEND DEPENDENCIES lingeling_project)
ExternalProject_ADD(lingeling_project
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/lingeling"
  CONFIGURE_COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/external/lingeling/configure"
  BUILD_COMMAND "make"
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND ""
  )

add_library(lingeling STATIC IMPORTED GLOBAL)
set_target_properties(lingeling PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/external/lingeling/liblgl.a")
target_include_directories(DD_ILP INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/external/lingeling/")

option(WITH_GUROBI OFF)
if(WITH_GUROBI)
  find_package(GUROBI REQUIRED)
  target_link_libraries(DD_ILP INTERFACE GUROBI)
  target_include_directories(DD_ILP INTERFACE "${GUROBI_INCLUDE_DIR}")
  target_compile_definitions(DD_ILP INTERFACE DD_ILP_WITH_GUROBI=1)
endif()

add_subdirectory(test) 
