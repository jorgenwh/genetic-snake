#pragma once

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "la.h"

py::list evaluate_population(std::vector<std::vector<py::array_t<float>>> &population, int size, int num_threads);
