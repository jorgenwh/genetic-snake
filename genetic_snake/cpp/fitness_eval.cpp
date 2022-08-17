#include <iostream>
#include <locale>
#include <iomanip>
#include <string>
#include <vector>
#include <tuple>
#include <thread>
#include <mutex>
#include <assert.h>
#include <chrono>
#include <stdio.h>

#include "snake_env.h"
#include "fitness_eval.h"

static std::mutex fetch_ind_mu;
static std::mutex dump_ind_mu;

static int ind_cntr;
static int evaluated_ind_cntr;
static int pop_size;
static std::string pop_size_formatted;
static int snake_size;

static std::vector<std::vector<py::array_t<float>>> st_population;
static std::vector<std::tuple<int, int>> game_results;

static int highscore; 
static float mean_score;
static float mean_fitness;

std::string format_with_commas(int value) {
  std::string s = std::to_string(value);
  int n = s.length() - 3;
  int end = (value >= 0) ? 0 : 1;
  while (n > end) {
    s.insert(n, ",");
    n -= 3;
  }
  return s;
}

std::vector<size_t> get_np_shape(py::array_t<float> &arr) {
  size_t ndims = arr.ndims();
  std::vector<size_t> shape(ndims);
  for (size_t i = 0; i < ndims; i++) {
    shape[i] = arr.shape()[i];
  }
  return shape;
}

int fetch_next_ind() {
  std::lock_guard<std::mutex> lock(fetch_ind_mu);

  if (ind_cntr >= pop_size) {
    return -1;
  }
  else {
    ind_cntr++;
    return ind_cntr-1;
  }
}

void dump_ind() {
  std::lock_guard<std::mutex> lock(dump_ind_mu);

  evaluated_ind_cntr++;
  std::string ind_cntr_formatted = format_with_commas(evaluated_ind_cntr-1);
  std::cout << "\33[3mIndividual\33[0m    : \33[1m" << ind_cntr_formatted << "\33[0m / \33[1m" << pop_size_formatted << "\33[0m\r";
}

void run_evaluation_game(std::vector<py::array_t<float>> &params, SnakeEnv &env, int &steps, int &score) {
  /*py::module_ np = py::module_::import("numpy");
  py::array_t<float> h = env.reset();
  py::array_t<float> zh;
  int num_params = params.size();
  while (!env.is_terminal()) {
    // Forward
    for (int i = 0; i < num_params; i++) {
      std::cout << h.shape() << std::endl;
      zh = np.attr("dot")(h, params[i]);
      if (i < num_params-1) {
        h = np.attr("maximum")(zh, 0);
      }
      else {
        h = zh;
      }
    }
    int action = np.attr("argmax")(h).cast<int>();
    std::cout << action << std::endl;

    h = env.step(action);
  }*/


  steps = env.steps;
  score = env.score;
  steps = 10;
  score = 1;
}

void thread_worker(int thread_id) {
  SnakeEnv env(snake_size);

  int ind_idx;
  std::vector<py::array_t<float>> params;
  while (true) {
    ind_idx = fetch_next_ind();

    if (ind_idx < 0) { break; }

    params = st_population[ind_idx];
    assert(params.size() == 3);

    int steps, score;
    run_evaluation_game(params, env, steps, score);

    game_results[ind_idx] = std::make_tuple(steps, score);
    mean_score += score;
    highscore = std::max(highscore, score);
    
    dump_ind();
  }
}

py::list evaluate_population(std::vector<std::vector<py::array_t<float>>> &population, int size, int num_threads) {
  ind_cntr = 0;
  evaluated_ind_cntr = 1;
  pop_size = population.size();
  pop_size_formatted = format_with_commas(pop_size);
  snake_size = size;
  highscore = 0;
  mean_score = 0;
  st_population = population;
  game_results.resize(pop_size);

  int thread_ids[num_threads];
  std::vector<std::thread> threads(num_threads);
  for (int i = 0; i < num_threads; i++) {
    thread_ids[i] = i;
    threads[i] = std::thread(&thread_worker, std::ref(thread_ids[i]));
  }
  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  std::cout << "\33[3mIndividual\33[0m    : \33[1m" << pop_size_formatted << "\33[0m / \33[1m" << pop_size_formatted << "\33[0m" << std::endl;

  py::list res;
  for (std::tuple<int, int> &t : game_results) {
    res.append(py::make_tuple(std::get<0>(t), std::get<1>(t)));
  }

  return res;
}
