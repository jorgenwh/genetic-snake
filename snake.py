from settings import settings
from miscellaneous import Point
from collections import deque
import numpy as np
import random

class Snake:
    def __init__(self, map_size=10, max_moves=100, network=None, game=None):
        self.map_size = map_size
        self.network = network
        self.deterministic = False

        # Action state and other details
        self.action_space = {
            'up': 0,
            'down': 1,
            'left': 2,
            'right': 3
        }
        self.direction = random.choice(list(self.action_space))
        self.tail_direction = self.direction
        self.is_alive = True
        self.has_won = False

        if game:
            self.game = game
            self.deterministic = True
            self.food_spawns = self.game['food_spawns']
            self.food_spawns = [(spawn[0], spawn[1]) for spawn in self.food_spawns]
            self.directions = self.game['directions']
            
            self.direction = self.directions.pop(0)

        # Current state vector representing the current game situation
        self.state_vector = None

        # The snake sees in 8 directions around its head
        self.vision_directions = [
            (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
        ]

        # Game details
        self.score = 0
        self.frames = 0
        self.frames_since_food = 0
        self.max_moves = max_moves

        # Initialize snake object
        self.snake_body = deque()
        self.init_snake()
        self.food_pos = self._create_food()

        # Update the snake object
        self.update_state_vector()
        self.network.forward(self.state_vector)

        # Store game data (and potentially store it) to replay the game later if the used neural network is loaded with it
        self.game_data = {'directions': [self.direction], 'food_spawns': [self.food_pos], 'start_position': self.snake_body[0]}

    def init_snake(self) -> None:
        if self.deterministic:
            head = self.game['start_position']  
        else:
            head = (random.randint(2, self.map_size - 3), random.randint(2, self.map_size - 3))

        if self.direction == 'up':
            body = [head, (head[0], head[1] + 1), (head[0], head[1] + 2)]
        elif self.direction == 'down':
            body = [head, (head[0], head[1] - 1), (head[0], head[1] - 2)]
        elif self.direction == 'left':
            body = [head, (head[0] + 1, head[1]), (head[0] + 2, head[1])]
        elif self.direction == 'right':
            body = [head, (head[0] - 1, head[1]), (head[0] - 2, head[1])]

        self.snake_body = deque(body)

    def update(self) -> None:
        self.update_state_vector()
        self.network.forward(self.state_vector)
        action = np.argmax(self.network.activations[-1])
        
        self.change_direction(action)

    def _create_food(self) -> tuple:
        # If we are replaying a game
        if self.deterministic:
            return self.food_spawns.pop(0)

        positions = []
        for x in range(self.map_size):
            for y in range(self.map_size):
                if (x, y) not in self.snake_body:
                    positions.append((x, y))
        return random.choice(positions)

    
    def change_direction(self, action: int) -> None:
        if self.deterministic:
            direction = self.directions.pop(0)
        else:
            for key in self.action_space:
                if self.action_space[key] == action:
                    direction = key
            self.game_data['directions'].append(direction)
        
        if not self._is_backwards(direction):
            self.direction = direction

    def move(self) -> None:
        if not self.is_alive:
            return

        head = self.snake_body[0]
        if self.direction == 'up':
            next_pos = (head[0], head[1] - 1)
        elif self.direction == 'down':
            next_pos = (head[0], head[1] + 1)
        elif self.direction == 'left':
            next_pos = (head[0] - 1, head[1])
        elif self.direction == 'right':
            next_pos = (head[0] + 1, head[1])

        if self._is_valid(next_pos):
            if next_pos == self.snake_body[-1]:
            # If we are moving into the previous position of the tail
                self.snake_body.pop()
                self.snake_body.appendleft(next_pos)
                
            elif next_pos == self.food_pos:
            # If we are eating a food we don't remove the tail
                self.score += 1
                self.frames_since_food = 0
                self.snake_body.appendleft(next_pos)
                if len(self.snake_body) != self.map_size**2:
                    self.food_pos = self._create_food()
                    self.game_data['food_spawns'].append(self.food_pos)
                else:
                    self.has_won = True
                
            else:
            # Otherwise we move normally
                self.snake_body.appendleft(next_pos)
                self.snake_body.pop()

            # Update the tail direction 
            tail = self.snake_body[-1]
            next_tail = self.snake_body[-2]
            x_diff = next_tail[0] - tail[0]
            y_diff = next_tail[1] - tail[1]

            if y_diff < 0:
                self.tail_direction = 'up'
            elif y_diff > 0:
                self.tail_direction = 'down'
            elif x_diff < 0:
                self.tail_direction = 'left'
            elif x_diff > 0:
                self.tail_direction = 'right'

            # If the snake has gone too long without eating a food it dies
            self.frames_since_food += 1
            if self.frames_since_food > self.max_moves:
                self.is_alive = False
            self.frames += 1

        else:
        # Otherwise it dies
            self.is_alive = False

    def _is_valid(self, position: tuple) -> bool:
    # If position is outside of the map it is invalid
        if not self._within_map(position):
            return False
        
        # We allow it to move into the tail position because the tail will be moved off
        if position == self.snake_body[-1]:
            return True

        # If the position is inside any other part of the snake body it is invalid
        if position in self.snake_body:
            return False
        
        return True

    def _is_backwards(self, action: str) -> bool:
    # If the given action means moving backwards into the snake body, it is an illegal move
        if action == 'up' and self.direction == 'down':
            return True
        if action == 'down' and self.direction == 'up':
            return True
        if action == 'right' and self.direction == 'left':
            return True
        if action == 'left' and self.direction == 'right':
            return True
        return False

    def update_state_vector(self) -> None:
    # Get the game state as input for the neural network to decide on a move
        vision = np.zeros((8, 3))

        head_pos = self.snake_body[0]
        for i, direction in enumerate(self.vision_directions):
            see_food, see_body = 0, 0
            cur_pos = head_pos
            steps_looked = 0

            while self._within_map(cur_pos):
                steps_looked += 1
                cur_pos = (cur_pos[0] + direction[0], cur_pos[1] + direction[1])
                if self.food_pos == cur_pos:
                    see_food = 1
                if cur_pos in self.snake_body:
                    see_body = 1    
            distance_to_wall = 1.0 / steps_looked

            vision[i][0] = distance_to_wall
            vision[i][1] = see_food
            vision[i][2] = see_body
        
        direction_vector = self._get_direction_vector()

        vision = vision.reshape(vision.shape[0] * vision.shape[1])
        self.state_vector = np.concatenate((vision, direction_vector)).reshape(1, 32)

    def _within_map(self, position: tuple) -> bool:
    # Check whether a given position is within the map
        x, y = position
        if x < 0 or x > self.map_size - 1 or y < 0 or y > self.map_size - 1:
            return False
        return True

    def _get_direction_vector(self) -> np.ndarray:
    # One hot vector containing snake and tail directions
        one_hot = np.zeros(8)
        one_hot[self.action_space[self.direction]] = 1
        one_hot[self.action_space[self.tail_direction]+4] = 1
        return one_hot