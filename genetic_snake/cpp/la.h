#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

py::array_t<float> matmul(py::array_t<float> &arr1, py::array_t<float> &arr2);
void relu(py::array_t<float> &arr);
int argmax(py::array_t<float> &arr);
