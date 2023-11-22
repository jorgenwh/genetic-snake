#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "snake_env.h"
#include "fitness_eval.h"

namespace py = pybind11;

PYBIND11_MODULE(genetic_snake_C, m) {
  m.doc() = "genetic_snake_C module";

  py::class_<SnakeEnv>(
      m, "SnakeEnv"
  )
    .def(py::init<int>())
    .def("step", &SnakeEnv::step, py::return_value_policy::take_ownership)
    .def("reset", &SnakeEnv::reset, py::return_value_policy::take_ownership)
    .def("is_terminal", &SnakeEnv::is_terminal)
    .def("print", &SnakeEnv::print)
    .def("get_snake_body", &SnakeEnv::get_snake_body)
    .def("get_food", &SnakeEnv::get_food)
    .def_readwrite("steps", &SnakeEnv::steps)
    .def_readwrite("score", &SnakeEnv::score)
    ;

  m.def("evaluate_population", &evaluate_population);
}
