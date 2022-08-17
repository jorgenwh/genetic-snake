#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <deque>
#include <random>
#include <algorithm>

enum flags {
  FLAG_ALIVE,
  FLAG_DEAD,
  FLAG_WON
};

struct coord {
  int x, y;
  
  bool operator==(const coord &other) const {
    return (x == other.x && y == other.y);
  }
  coord &operator=(const coord &other) {
    x = other.x;
    y = other.y;
    return *this;
  }
};

extern coord directions[4];
extern coord vision_directions[8];

class SnakeEnv {
public:
  SnakeEnv(int size);
  ~SnakeEnv();

  py::array_t<float> step(int action);
  py::array_t<float> reset();
  float *step_raw(int action);
  float *reset_raw();
  bool is_terminal();

  void print();

  int size;
  int score = 0;
  int steps = 0;

  std::deque<coord> snake;
  coord food;

  py::list get_snake_body();
  py::tuple get_food();
private:
  int steps_since_food = 0;

  int win_score;

  int head_direction;
  int tail_direction;

  int flag = FLAG_ALIVE;

  std::mt19937 *rng;
  std::uniform_int_distribution<std::mt19937::result_type> *dist;

  void initialize_snake();
  void set_food();

  float *get_state();
  inline bool is_within_map(coord &c);
  inline bool is_valid(coord &c);
  inline bool is_backwards(int action);
};
