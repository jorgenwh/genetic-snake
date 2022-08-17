cmake_minimum_required(VERSION 3.16)
project(cpp_module)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3 -pthread")

add_subdirectory(extern/pybind11)

set(SOURCES ${SOURCES}
  genetic_snake/cpp/snake_env.cpp
  genetic_snake/cpp/fitness_eval.cpp
  genetic_snake/cpp/la.cpp
)

set(BINDINGS
  genetic_snake/cpp/module.cpp
)

add_library(cpp_module MODULE "${SOURCES}" "${BINDINGS}")
target_link_libraries(cpp_module PRIVATE pybind11::module)
set_target_properties(cpp_module PROPERTIES
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}")
