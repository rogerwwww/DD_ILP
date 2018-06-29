project(DD_ILP)

cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# C++14
add_compile_options(-std=c++14)

# compiler options
add_definitions(-march=native)

add_subdirectory(external/lingeling)

add_library(DD_ILP INTERFACE)
target_include_directories(DD_ILP INTERFACE . include )
target_link_libraries(DD_ILP INTERFACE lingeling)


option(WITH_GUROBI OFF)
if(WITH_GUROBI)
  find_package(GUROBI REQUIRED)
  target_link_libraries(DD_ILP INTERFACE GUROBI)
  target_include_directories(DD_ILP INTERFACE "${GUROBI_INCLUDE_DIR}")
  target_compile_definitions(DD_ILP INTERFACE DD_ILP_WITH_GUROBI=1)
endif()

add_subdirectory(test) 

option(BUILD_DOC "Build documentation" OFF)
if(BUILD_DOC)
  find_package(Doxygen REQUIRED)
  configure_file(Doxyfile.in Doxyfile @ONLY)
  add_custom_target(docs-ddilp
    COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)
  if(NOT TARGET docs)
    add_custom_target(docs ALL)
  endif()
  add_dependencies(docs docs-ddilp)
endif()
