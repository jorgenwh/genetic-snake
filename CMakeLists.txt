cmake_minimum_required(VERSION 3.16)
project(genetic_snake LANGUAGES CXX)

find_package(pybind11 REQUIRED)

set(CMAKE_CXX_FLAGS "-O3 -pthread")

add_subdirectory("genetic_snake/_C")
