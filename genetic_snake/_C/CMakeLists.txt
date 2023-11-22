cmake_minimum_required(VERSION 3.16)
project(genetic_snake_C LANGUAGES CXX)

set(SOURCES
    fitness_eval.cpp
    snake_env.cpp
)

pybind11_add_module(${PROJECT_NAME} ${SOURCES} "bindings.cpp")
