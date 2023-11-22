#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <thread>
#include <mutex>

#include "snake_env.h"
#include "fitness_eval.h"

static std::mutex mu;
static int ind_counter;
static std::string population_size_formatted;

static std::string format_with_commas(int v) {
  std::string s = std::to_string(v);
  int n = s.length() - 3;
  int end = (v>= 0) ? 0 : 1;
  while (n > end) {
    s.insert(n, ",");
    n -= 3;
  }
  return s;
}

static void dump_ind() {
  std::lock_guard<std::mutex> lock(mu);
  ind_counter++;
  std::string ind_counter_formatted = format_with_commas(ind_counter);
  std::cout 
    << "\33[3mIndividual\33[0m    : \33[1m" 
    << ind_counter_formatted << "\33[0m / \33[1m" 
    << population_size_formatted << "\33[0m\r";
}

inline static std::vector<int> get_np_shape(py::array_t<float> &arr) {
  int ndim = arr.ndim();
  std::vector<int > shape(ndim);

  for (int i = 0; i < ndim; i++) {
    shape[i] = arr.shape()[i];
  }

  return shape;
}

inline static void matmul(const float *A, const float *B, float *C, 
    const int Ac, const int Bc, const int Cr, const int Cc) {
  for (int i = 0; i < Cr; i++) {
    for (int j = 0; j < Cc; j++) {
      for (int k = 0; k < Ac; k++) {
        C[i*Cc + j] += A[i*Ac + k] * B[k*Bc + j];
      }
    }
  }
}

inline static void relu(float *arr, size_t size) {
  for (size_t i = 0; i < size; i++) {
    arr[i] = (arr[i] > 0) ? arr[i] : 0.0f;
  }
}

inline static int argmax(float *arr, int size) {
  float max = -std::numeric_limits<float>::max();
  int action = 0;

  for (int i = 0; i < size; i++) {
    if (arr[i] > max) {
      action = i;
      max = arr[i];
    }
  }

  return action;
}

void thread_worker(
    const int &offset, 
    const int &num_threads,
    const int &snake_size,
    std::vector<std::vector<py::array_t<float>>> &population, 
    std::vector<std::tuple<int, int>> &game_results) {
  SnakeEnv env(snake_size);

  float *h1 = new float[20];
  float *h2 = new float[12];
  float *h3 = new float[4];

  for (size_t i = offset; i < population.size(); i+=num_threads) {
    std::vector<py::array_t<float>> ind = population[i];
    
    const float *w1 = ind[0].data(); 
    const float *w2 = ind[1].data(); 
    const float *w3 = ind[2].data(); 

    float *observation = env.reset_raw();
    while (!env.is_terminal()) {
      memset(h1, 0, sizeof(float)*20);
      memset(h2, 0, sizeof(float)*12);
      memset(h3, 0, sizeof(float)*4);

      // Get action by forwarding through network
      matmul(observation, w1, h1, 32, 20, 1, 20);
      relu(h1, 20);
      matmul(h1, w2, h2, 12, 12, 1, 12);
      relu(h2, 12);
      matmul(h2, w3, h3, 12, 4, 1, 4);

      //int action = 1;
      int action = argmax(h3, 4); 
      
      // Free previous observation
      delete[] observation;

      float *observation = env.step_raw(action);
    }

    delete[] observation;

    std::get<0>(game_results[i]) = env.steps;
    std::get<1>(game_results[i]) = env.score;

    dump_ind();
  }

  delete[] h1;
  delete[] h2;
  delete[] h3;
}

py::list evaluate_population(
    std::vector<std::vector<py::array_t<float>>> &population, 
    const int snake_size, const int num_threads) {
  const size_t population_size = population.size();
  std::vector<std::tuple<int,int>> game_results(population_size);

  ind_counter = 0;
  population_size_formatted = format_with_commas(population_size);

  std::vector<std::thread> threads(num_threads);
  int thread_offsets[num_threads];
  for (int i = 0; i < num_threads; i++) {
    thread_offsets[i] = i;
    threads[i] = std::thread(
        &thread_worker, 
        std::ref(thread_offsets[i]),
        std::ref(num_threads),
        std::ref(snake_size),
        std::ref(population),
        std::ref(game_results));
  }
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  std::string population_size_formatted = format_with_commas((int)population_size);
  std::cout 
    << "\33[3mIndividual\33[0m    : \33[1m" 
    << population_size_formatted << "\33[0m / \33[1m" 
    << population_size_formatted << "\33[0m\n";

  return py::cast(game_results);
}

