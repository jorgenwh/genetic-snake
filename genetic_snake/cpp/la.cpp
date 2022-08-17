#include <limits>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "la.h"

py::array_t<float> matmul(py::array_t<float> &arr1, py::array_t<float> &arr2) {
  assert(arr1.ndim() == 2);
  assert(arr2.ndim() == 2);

  size_t arr1_rows = arr1.shape()[0];
  size_t arr1_cols = arr1.shape()[1];
  size_t arr2_rows = arr2.shape()[0];
  size_t arr2_cols = arr2.shape()[1];
  size_t out_rows = arr1_rows;
  size_t out_cols= arr2_cols;

  assert(arr1_cols == arr2_rows);

  py::array_t<float> out_arr = py::array_t<float>({arr1_rows, arr2_cols});

  float *out = (float *)out_arr.mutable_data();
  float *in1 = (float *)arr1.mutable_data();
  float *in2 = (float *)arr2.mutable_data();

  memset(out, 0, sizeof(float)*out_rows*out_cols);

  for (size_t i = 0; i < out_rows; i++) {
    for (size_t j = 0; j < out_cols; j++) {
      for (size_t k = 0; k < arr1_cols; k++) {
        out[i*out_cols + j] += in1[i*arr1_cols + k] * in2[k*arr2_cols + j];
      }
    }
  }

  return out_arr;
}

void relu(py::array_t<float> &arr) {
  size_t size = arr.size();
  float *data = (float *)arr.mutable_data();

  for (size_t i = 0; i < size; i++) {
    data[i] = (data[i] > 0) ? data[i] : 0.0f;
  }
}

// Only works for vectors
int argmax(py::array_t<float> &arr) {
  assert(arr.ndim() == 2);
  assert(arr.shape()[0] == 1);
  size_t size = arr.size();

  int action = 0;
  float max = -std::numeric_limits<float>::max();

  const float *data = (float *)arr.data();
  for (int i = 0; i < size; i++) {
    if (data[i] > max) {
      max = data[i];
      action = i;
    }
  }

  return action;
}
