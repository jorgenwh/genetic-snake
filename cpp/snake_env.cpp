#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <deque>
#include <random>
#include <algorithm>
#include <iostream>
#include <assert.h>

#include "snake_env.h"

coord directions[4] = {
  {0, -1}, // NORTH
  {1, 0},  // EAST
  {0, 1},  // SOUTH
  {-1, 0}, // WEST
};

coord vision_directions[8] = {
  {0, -1}, // NORTH
  {1, -1}, // NORTH-EAST 
  {1, 0},  // EAST 
  {1, 1},  // SOUTH-EAST 
  {0, 1},  // SOUTH
  {-1, 1}, // SOUTH-WEST
  {-1, 0}, // WEST
  {-1, -1} // NORTH-WEST
};

SnakeEnv::SnakeEnv(int size) {
  this->size = size;
  win_score = std::pow(size, 2) - 3;

  std::random_device rdev;
  rng = new std::mt19937(rdev());
  dist = new std::uniform_int_distribution<std::mt19937::result_type>(0, size - 1);

  head_direction = ((*dist)(*rng)) % 4;
  tail_direction = head_direction;

  initialize_snake();
  set_food();
}

SnakeEnv::~SnakeEnv() {
  delete rng;
  delete dist;
}

py::tuple SnakeEnv::step(int action) {
  if (is_backwards(action)) { action = head_direction; }

  head_direction = action;

  coord new_head_pos = {(*snake.begin()).x + directions[action].x, (*snake.begin()).y + directions[action].y};

  if (is_valid(new_head_pos)) {
    snake.push_front(new_head_pos);

    if (new_head_pos == food) {
      score++;
      steps_since_food = 0;
      set_food();
      
      if (score == win_score) {
        flag = FLAG_WON;
      }
    }
    else {
      snake.pop_back();
      steps_since_food++;

      if (steps_since_food > std::pow(size, 2)) {
        flag = FLAG_DEAD;
      }
    }

    coord tail = *snake.end();
    coord next_tail = *(snake.end() - 1);
    int x_diff = next_tail.x - tail.x;
    int y_diff = next_tail.y - tail.y;

    if (y_diff < 0) { tail_direction = 0; }
    else if (y_diff > 0) { tail_direction = 2; }
    else if (x_diff < 0) { tail_direction = 3; }
    else if (x_diff > 0) { tail_direction = 1; }
  }
  else {
    flag = FLAG_DEAD;
  }

  steps++;

  float *state_data = get_state();
  py::array_t<float> state = py::array_t<float>({1, 32}, state_data);
  delete[] state_data;
  bool done = (flag != FLAG_ALIVE);

  return py::make_tuple(state, done);
}

py::array_t<float> SnakeEnv::reset() {
  head_direction = ((*dist)(*rng)) % 4;
  assert(head_direction >= 0 && head_direction < 4);

  tail_direction = head_direction;

  initialize_snake();
  set_food();

  score = 0;
  steps = 0;
  steps_since_food = 0;
  flag = FLAG_ALIVE;

  float *state_data = get_state();
  py::array_t<float> state = py::array_t<float>({1, 32}, state_data);
  delete[] state_data;
  return state;
}

bool SnakeEnv::is_terminal() {
  return (flag != FLAG_ALIVE);
}

py::list SnakeEnv::get_snake_body() {
  std::vector<py::tuple> body(snake.size());

  for (int i = 0; i < snake.size(); i++) {
    body[i] = py::make_tuple(snake[i].x, snake[i].y);
  }

  return py::cast(body);
}

py::tuple SnakeEnv::get_food() {
  return py::make_tuple(food.x, food.y);
}

void SnakeEnv::print() {
  for (int x = 1; x < size + 1; x++) {
    std::cout << x << " ";
  }
  std::cout << "\n";
  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      coord cur = {x, y};
      std::string sym = "-";

      if (cur == *snake.begin()) { sym = "\33[1m\33[92mh\33[0m"; }
      else if (std::find(snake.begin(), snake.end(), cur) != snake.end()) { sym = "\33[1m\33[92mx\33[0m"; }
      else if (cur == food) { sym = "\33[1m\33[91mo\33[0m"; }
        
      std::cout << sym << " ";
    }
  std::cout << "\n";
  }
}

void SnakeEnv::initialize_snake() {
  if (!snake.empty()) { snake.clear(); }

  coord head;
  head.x = (((*dist)(*rng)) & (size - 4)) + 2;
  head.y = (((*dist)(*rng)) & (size - 4)) + 2;
  snake.push_back(head);

  coord dir = directions[head_direction];
  coord b1, b2;
  b1.x = head.x - dir.x;
  b1.y = head.y - dir.y;
  b2.x = head.x - dir.x*2;
  b2.y = head.y - dir.y*2;
  snake.push_back(b1);
  snake.push_back(b2);
}

void SnakeEnv::set_food() {
  std::vector<coord> valid_positions;
  for (int x = 0; x < size; x++) {
    for (int y = 0; y < size; y++) {
      coord cur;
      cur.x = x;
      cur.y = y;
      if (std::find(snake.begin(), snake.end(), cur) == snake.end()) {
        valid_positions.push_back(cur);
      }
    }
  }

  auto start = valid_positions.begin();
  auto end = valid_positions.end();
  std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
  std::advance(start, dis(*rng));
  food = *start;
}

float *SnakeEnv::get_state() {
  float *data = new float[32];
  memset(data, 0, sizeof(float)*32);

  coord head = *snake.begin();

  for (int i = 0; i < 8; i++) {
    float _food = 0.0f, _body = 0.0f, steps_looked = 0.0f;
    coord cur = head;

    while (is_within_map(cur)) {
      steps_looked++;
      cur.x = cur.x + vision_directions[i].x;
      cur.y = cur.y + vision_directions[i].y;

      if (food == cur) { _food = 1.0f; }
      if (std::find(snake.begin(), snake.end(), cur) != snake.end()) { _body = 1.0f; }
    }

    data[i*3] = 1.0f / steps_looked;
    data[i*3 + 1] = _food;
    data[i*3 + 2] = _body;
  }

  data[24 + head_direction] = 1.0f;
  data[28 + tail_direction] = 1.0f;

  return data;
}

inline bool SnakeEnv::is_within_map(coord &c) {
  return (c.x >= 0 && c.y >= 0 && c.x < size && c.y < size);
}

inline bool SnakeEnv::is_valid(coord &c) {
  return (is_within_map(c) && std::find(snake.begin(), snake.end(), c) == snake.end());
}

inline bool SnakeEnv::is_backwards(int action) {
  return std::abs(action - head_direction) == 2;
}
