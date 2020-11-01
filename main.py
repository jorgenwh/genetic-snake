from settings import settings
from miscellaneous import Point
from storage import save_model, load_model, save_game, load_game
from genetic_functions import elitist_selection, roulette_wheel_selection
from genetic_functions import crossover, mutate
from genetic_functions import single_point_crossover, simulated_binary_crossover
from genetic_functions import gaussian_mutation
from snake_widget import SnakeWidget
from neural_network_widget import NeuralNetWidget
from text_widget import TextWidget
from snake import Snake
from neural_network import NeuralNetwork
from individual import Individual
from PyQt5 import QtGui, QtCore, QtWidgets
from collections import deque
import numpy as np
import random
import time
import sys

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings):
        super().__init__()
        self.setAutoFillBackground(True)
        
        # Settings
        self.settings = settings

        # Application settings
        self.fps = self.settings['fps']
        self.update_window = self.settings['update_window']
        self.save_models = self.settings['save_best_models']
        self.save_game = self.settings['save_best_game']
        self.load_model = self.settings['load_model']
        self.load_game = self.settings['load_game']
        self.load_game_size = self.settings['load_game_size']
        self.window_title = self.settings['window_title']
        assert not (self.load_model and self.load_game), 'Cannot load a network and a game at the same time as network will be overwritten'

        # Algorithm specifications
        self.n_parents = self.settings['n_parents']
        self.n_children = self.settings['n_children']
        self.population_size = self.n_parents + self.n_children
        self.max_generations = self.settings['max_generations']
        self.max_moves = self.settings['max_moves']
        self.neuralnet_structure = self.settings['network_structure']

        # Game grid specifications
        self.map_size = self.settings['map_size']
        self.cell_size = self.settings['cell_size']
        self.draw_grid = self.settings['draw_grid']

        # Neural network visual specifications
        self.layer_offset = self.settings['layer_offset']
        self.neuron_offset = self.settings['neuron_offset']
        self.neuron_radius = self.settings['neuron_radius']

        # Initialize population of individuals, each with its own neural network to act
        self.population = [
            Individual(NeuralNetwork(layer_dim=self.neuralnet_structure)) for _ in range(self.population_size)
            ]
        self.cur_ind = 0
        self.individual = self.population[self.cur_ind]

        if self.load_model:
        # If we are only showing a saved model
            self.neuralnet_structure, weights = load_model(self.map_size)
            self.individual = Individual(
                NeuralNetwork(
                    self.neuralnet_structure,
                    weights
                ))
            self.save_models = False
            self.save_game = False
            self.update_window = True
            self.max_moves = np.inf

        elif self.load_game:
        # If we are replaying a stored game
            self.game_data = load_game(self.map_size)
            self.save_models = False
            self.save_game = False
            self.update_window = True
            self.max_moves = np.inf
        
        # Create a snake
        if self.load_game:
            self.snake = Snake(self.map_size, self.max_moves, self.individual.network, game=self.game_data)
        else:
            self.snake = Snake(self.map_size, self.max_moves, self.individual.network, game=None)

        # Statistics
        self.generation = 1
        self.highest_score = 0
        self.average_score = 0
        self.average_fitness = 0
        self.average_fitness_scores = deque(maxlen=10)
        self.new_highest = False

        # Origin of where the application window appears
        self.window_origin = Point(80, 50)

        # Size of the snake game widget
        self.snake_widget_width = self.map_size * self.cell_size
        self.snake_widget_height = self.map_size * self.cell_size

        # Size of the neural network widget
        self.neuralnet_widget_width = (len(self.neuralnet_structure) * self.layer_offset) - 40
        self.neuralnet_widget_height = max(self.neuralnet_structure) * ((self.neuron_radius * 2) + self.neuron_offset)

        # Size of the text widget
        self.text_widget_width = self.snake_widget_width
        self.text_widget_height = 255

        # Application window size
        self.window_width = 85 + self.snake_widget_width + self.neuralnet_widget_width
        self.window_height = 80 + max(self.neuralnet_widget_height, self.snake_widget_height + self.text_widget_height)

        # Initialize the application window with its widgets
        self.init_application_window(self.window_title)

        # Timer to call on update every 1000 / fps ms
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000 / self.fps)

        # Show application window
        self.show()

    def init_application_window(self, window_title: str) -> None:
        """Initialize the application window and all its widgets"""
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle(window_title)
        self.setGeometry(
            self.window_origin.x, self.window_origin.y, 
            self.window_width, self.window_height)

        # Create snake game widget
        self.snake_widget = SnakeWidget(
            self.centralWidget, self.map_size, self.cell_size, self.draw_grid, self.snake)
        self.snake_widget.setGeometry(35, 50, self.snake_widget_width, self.snake_widget_height)

        # Create neural network widget
        self.network_widget = NeuralNetWidget(self.centralWidget, self.individual.network)
        self.network_widget.setGeometry(
            110 + self.snake_widget_width, 50, 
            self.neuralnet_widget_width, self.neuralnet_widget_height)

        # Create text widget
        self.text_widget = TextWidget(self.centralWidget, self.population_size)
        self.text_widget.setGeometry(
            35, 50 + self.snake_widget_height, self.text_widget_width, self.text_widget_height)

    def update(self) -> None:
        self.snake_widget.update(self.update_window, self.draw_grid)
        self.network_widget.update(self.update_window)
        self.text_widget.update(
            self.update_window, self.cur_ind, self.generation, self.snake.score, self.highest_score, self.average_fitness)

        if self.snake.is_alive:
            self.snake.move()
            if self.snake.score > self.highest_score:
                self.highest_score = self.snake.score
                self.new_highest = True

            # If we are processing without visuals, update the visuals to show the completed game
            if self.snake.has_won:
                self.update_window = True

        # If an agent has won the game, we store the game and model
        elif self.snake.has_won:
            print(f"ind {self.cur_ind} | gen: {self.generation} | top score: {self.highest_score} | avg score: {round(self.average_score, 2)}     \r", end='')
            if self.save_game:
                save_game(self.snake.game_data, self.map_size)
                save_model(self.individual.network.weights, self.neuralnet_structure, self.map_size)
            self.snake.is_alive = False
            self.snake.has_won = False

        else:
            # If we were showcasing the best stored neural net or game
            if self.load_model or self.load_game:
                print(f"score: {self.highest_score}")
                self.timer.stop()
                return

            if self.save_models and self.new_highest:
                save_model(self.individual.network.weights, self.neuralnet_structure, self.map_size)
            if self.save_game and self.new_highest:
                save_game(self.snake.game_data, self.map_size)
            self.new_highest = False

            # Finish collecting the data for the current individual
            self.individual.score = self.snake.score
            self.individual.steps = self.snake.frames
            self.individual.compute_fitness()
            self.cur_ind += 1

            print(f"ind {self.cur_ind} | gen: {self.generation} | top score: {self.highest_score} | avg score: {round(self.average_score, 2)}     \r", end='')
            
            if self.cur_ind != self.population_size:
            # We haven't evaluated all the networks in the population yet
                self.individual = self.population[self.cur_ind]
                self.snake = Snake(self.map_size, self.max_moves, self.individual.network)
                self.snake.update_state_vector()
                self.snake_widget.new_game(self.snake)
                self.network_widget.network = self.individual.network
                self.individual.network.forward(self.snake.state_vector)

            else:
            # We have evaluated all the individuals. GA stuff happens here
                scores = [ind.score for ind in self.population]
                self.average_score = sum(scores) / len(scores)

                fitness_scores = [ind.fitness for ind in self.population]
                self.average_fitness_scores.append(sum(fitness_scores) / self.population_size)
                self.average_fitness = sum(self.average_fitness_scores) / min(10.0, self.generation+1)

                self.process_generation()

                self.generation += 1
                self.cur_ind = 0
                self.individual = self.population[self.cur_ind]

                if self.generation >= self.max_generations+1:
                    quit()

    def process_generation(self) -> None:
        self.population = elitist_selection(self.population, self.n_parents)
        random.shuffle(self.population)

        children = []
        while len(children) < self.n_children:
            mom, dad = roulette_wheel_selection(self.population, 2)

            # Breed parents to create two children
            son, daughter = crossover(mom, dad)

            # Mutate the children
            mutate(son)
            mutate(daughter)

            children.extend([son, daughter])
        
        self.population.extend(children)
        random.shuffle(self.population)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
    # Handle input commands
        key_press = event.key()
        if key_press == QtCore.Qt.Key_S:
            if self.update_window:
                self.update_window = False
            else:
                self.update_window = True

        elif key_press == QtCore.Qt.Key_G:
            if self.draw_grid:
                self.draw_grid = False
            else:
                self.draw_grid = True

        elif key_press == QtCore.Qt.Key_E:
            self.timer.stop()
            quit(0)
        elif key_press == QtCore.Qt.Key_1:
            self.timer.setInterval(1000 / 5)
        elif key_press == QtCore.Qt.Key_2:
            self.timer.setInterval(1000 / 10)
        elif key_press == QtCore.Qt.Key_3:
            self.timer.setInterval(1000 / 200)
        elif key_press == QtCore.Qt.Key_4:
            self.timer.setInterval(1000 / 5000)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(settings)
    sys.exit(app.exec_())