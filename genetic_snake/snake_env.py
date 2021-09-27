import numpy as np
from collections import deque
from typing import Tuple, Deque

class SnakeEnv():
    """
    Class to store the state and simulation of a snake game environment.
    """
    def __init__(self, args):
        self.args = args

        """ 
        0 = up
        1 = right
        2 = down
        3 = left
        """
        self.head_direction = np.random.randint(4)
        self.tail_direction = self.head_direction
        self.terminal = False
        self.won = False

        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        
        self.snake_body = self.initialize_snake()
        self.food = self.create_food()

        self.directions = {
            0: (0, -1),
            1: (1, 0),
            2: (0, 1),
            3: (-1, 0)
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        # the snake will continue forward if the action is illegal (backwards)
        if self.backwards(action):
            action = self.head_direction

        self.head_direction = action

        new_head_position = (self.snake_body[0][0] + self.directions[action][0], self.snake_body[0][1] + self.directions[action][1])
        reward = -0.01

        # Check whether the action is valid, or if it kills the snake
        if self.valid(new_head_position):
            self.snake_body.appendleft(new_head_position)
            
            # Check if the action led to eating a food
            if new_head_position == self.food:
                # If the entire map has been filled, the player has won
                reward = 1
                self.score += 1
                self.steps_since_food = 0
                self.food = self.create_food()
                if self.score == (self.args.size ** 2) - 3:
                    self.terminal = self.won = True
            else:
                self.snake_body.pop()
                self.steps_since_food += 1

                # If step limit has been reached, the game ends without a win
                if self.steps_since_food > self.args.size ** 2:
                    self.terminal = True

            tail = self.snake_body[-1]
            next_tail = self.snake_body[-2]
            x_diff = next_tail[0] - tail[0]
            y_diff = next_tail[1] - tail[1]

            if y_diff < 0:
                self.tail_direction = 0
            elif y_diff > 0:
                self.tail_direction = 2
            elif x_diff < 0:
                self.tail_direction = 3
            elif x_diff > 0:
                self.tail_direction = 1

        # If the action killed the snake
        else:
            self.terminal = True
            reward = -1
        self.steps += 1

        # Return the game state observation, the reward for the last action and whether
        # the action ended the game
        return self.observe(), reward, self.terminal
        
    # Reset the game state to a starting position
    def reset(self) -> np.ndarray:
        self.head_direction = np.random.randint(4)
        self.tail_direction = self.head_direction
        self.terminal = False
        self.won = False

        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        
        self.snake_body = self.initialize_snake()
        self.food = self.create_food()

        return self.observe()
    
    # Check whether a position is valid (within the game map and not inside the snake body)
    def valid(self, position: Tuple[int, int]) -> bool:
        return self.within_map(position) and position not in self.snake_body
        
    # Check whether a position is within the map
    def within_map(self, position: Tuple[int, int]) -> bool:
        x, y = position
        return x >= 0 and x < self.args.size and y >= 0 and y < self.args.size

    # Check whether an action means moving backwards from the current direction (into the snake body)
    def backwards(self, action: int) -> bool:
        if action == 0:
            return self.head_direction == 2
        if action == 1:
            return self.head_direction == 3
        if action == 2:
            return self.head_direction == 0
        if action == 3:
            return self.head_direction == 1

    def initialize_snake(self) -> Deque[Tuple[int, int]]:
        head = (np.random.randint(2, self.args.size - 3), np.random.randint(2, self.args.size - 3))

        if self.head_direction == 0:
            snake_body = [head, (head[0], head[1] + 1), (head[0], head[1] + 2)]
        elif self.head_direction == 1:
            snake_body = [head, (head[0] - 1, head[1]), (head[0] - 2, head[1])]
        elif self.head_direction == 2:
            snake_body = [head, (head[0], head[1] - 1), (head[0], head[1] - 2)]
        else:
            snake_body = [head, (head[0] + 1, head[1]), (head[0] + 2, head[1])]
        
        return deque(snake_body)

    # Create a new (valid) position for the food
    def create_food(self) -> Tuple[int, int]:
        valid_positions = []
        for x in range(self.args.size):
            for y in range(self.args.size):
                if (x, y) not in self.snake_body:
                    valid_positions.append((x, y))
        return valid_positions[np.random.randint(0, len(valid_positions))]

    # Get the game state vector
    def observe(self) -> np.ndarray:
        vision = np.zeros((8, 3))
        head_position = self.snake_body[0]
        for i, direction in enumerate([(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]):
            food, body, steps_looked = 0, 0, 0
            cur_position = head_position

            while self.within_map(cur_position):
                steps_looked += 1
                cur_position = (cur_position[0] + direction[0], cur_position[1] + direction[1])
                if self.food == cur_position:
                    food = 1
                if cur_position in self.snake_body:
                    body = 1

            vision[i,0] = 1.0 / steps_looked
            vision[i,1] = food
            vision[i,2] = body

        direction = np.zeros(8)
        direction[self.head_direction] = 1
        direction[self.tail_direction + 4] = 1

        return np.concatenate((vision.reshape(-1), direction)).reshape(1, 32)