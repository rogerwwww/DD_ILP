project(DD_ILP)

cmake_minimum_required(VERSION 2.8.12)

# C++14
add_compile_options(-std=c++14)

# compiler options
add_definitions(-march=native)

include(ExternalProject)

list(APPEND DEPENDENCIES lingeling_project)
ExternalProject_ADD(lingeling_project
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/lingeling"
  CONFIGURE_COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/external/lingeling/configure"
  BUILD_COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/external/lingeling/make"
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND ""
  )

add_library(lingeling STATIC IMPORTED GLOBAL)
set_target_properties(lingeling PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/external/lingeling/liblgl.a")
set_target_properties(lingeling PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/external/lingeling/")

include_directories(.)
add_subdirectory(test) 
