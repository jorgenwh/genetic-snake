from PyQt5 import QtCore, QtWidgets
import numpy as np

from ..genetic.individual import Individual
from ..genetic.functional import elitist_selection, roulette_wheel_selection
from ..genetic.functional import crossover
from ..genetic.functional import mutate
from .snake_gui import SnakeGUI
from .network_gui import NetworkGUI
from .info_gui import InfoGUI

class GUIApplication(QtWidgets.QMainWindow):
    def __init__(self, args, snake_env_class, loaded_individual=None, output_dir_name=None):
        super().__init__()
        self.args = args
        self.snake_env_class = snake_env_class
        self.loaded = False
        self.output_dir_name = output_dir_name

        self.ind_idx = 0
        if loaded_individual is None:
            # Initialize a randomly generated population
            self.population = [Individual() for _ in range(self.args.parents + self.args.children)]

            self.individual = self.population[self.ind_idx]
        else:
            self.individual = loaded_individual
            self.loaded = True

        # Initialize the snake game environment simulation and get the initial observation
        self.snake_env = self.snake_env_class(self.args.size)
        self.observation = self.snake_env.reset()

        # Statistics
        self.generation = 1
        self.highscore = 0
        self.mean_score = 0
        self.mean_fitness = 0
        self.generation_score = 0

        self.activations = []

        # Flag for whether to render the gui or freeze it.
        # For significantly faster performance, turn rendering off.
        # This can be done inside the gui
        self.render = True

        # Variables to control fps (speed)
        self.fps_idx = 0 if self.loaded else 2
        self.fps_settings = [3, 10, 10000]
        self.fps = self.fps_settings[self.fps_idx]

        # Initialize the widgets
        self.initialize_gui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(1_000 / self.fps)
        self.show()

    def initialize_gui(self):
        snake_gui_size = self.args.size * 35, self.args.size * 35
        network_gui_size = (4 * 175) - 40, 32 * ((7.5 * 2) + 4.5)
        info_gui_size = snake_gui_size[0], 255
        width, height = 85 + snake_gui_size[0] + network_gui_size[0], 80 + max(network_gui_size[1], snake_gui_size[1] + info_gui_size[1], info_gui_size[1])

        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setGeometry(800, 300, width, height)
        self.setFixedSize(width, height)

        # snake game widget
        self.snake_gui = SnakeGUI(self.centralWidget, self.snake_env, self.args)
        self.snake_gui.setGeometry(35, 50, snake_gui_size[0], snake_gui_size[1])

        # neural network widget
        self.network_gui = NetworkGUI(self.centralWidget, [32, 20, 12, 4])
        self.network_gui.setGeometry(110 + snake_gui_size[0], 50, network_gui_size[0], network_gui_size[1])

        # text widget
        self.info_gui = InfoGUI(self.centralWidget, self.args)
        self.info_gui.setGeometry(35, 50 + snake_gui_size[0], info_gui_size[0], info_gui_size[1])

        print("\n\33[90m--------------------------------\33[0m\n\33[92mGeneration {:,}\33[0m".format(self.generation))

    def step(self):
        if self.snake_env.is_terminal():
            self.individual.compute_fitness(self.snake_env.score, self.snake_env.steps)
            self.generation_score += self.snake_env.score

            if self.loaded:
                return

            self.ind_idx += 1

            if self.snake_env.score == self.args.size ** 2 - 3:
                if not self.loaded:
                    if self.output_dir_name is not None:
                        self.individual.genome.save(self.output_dir_name)
                    else:
                        self.individual.genome.save(f"individual_gen_{self.generation}")
                exit(0)

            if self.ind_idx == self.args.parents + self.args.children:
                self.print_generation_summary()
                self.generation_step()
                print("\n\33[90m--------------------------------\33[0m\n\33[92mGeneration {:,}\33[0m".format(self.generation))
                print("\33[3mIndividual\33[0m    : \33[1m{:,}\33[0m / \33[1m{:,}\33[0m".format(self.ind_idx, self.args.parents + self.args.children), end="\r")
            else:
                self.individual = self.population[self.ind_idx]
                self.observation = self.snake_env.reset()
                print("\33[3mIndividual\33[0m    : \33[1m{:,}\33[0m / \33[1m{:,}\33[0m".format(self.ind_idx, self.args.parents + self.args.children), end="\r")

        else:
            self.activations = self.individual.act(self.observation)
            action = np.argmax(self.activations[-1])
            self.observation = self.snake_env.step(action)

            self.highscore = max(self.highscore, self.snake_env.score)

        if self.render:
            self.draw()

    def generation_step(self):
        self.mean_fitness = sum([individual.fitness for individual in self.population]) / (self.args.parents + self.args.children)
        self.mean_score = self.generation_score / (self.args.parents + self.args.children)
        self.generation_score = 0

        self.population = elitist_selection(self.population, self.args.parents)
        np.random.shuffle(self.population)

        children = []
        while len(children) < self.args.children:
            mom, dad = roulette_wheel_selection(self.population, 2)

            # breed
            son, daughter = crossover(mom, dad)

            # mutate children
            mutate(son)
            mutate(daughter)

            children.extend([son, daughter])

        self.population.extend(children)
        np.random.shuffle(self.population)

        self.generation += 1
        self.ind_idx = 0
        self.individual = self.population[self.ind_idx]

    def print_generation_summary(self):
        print("\33[3mIndividual\33[0m    : \33[1m{:,}\33[0m / \33[1m{:,}\33[0m".format(self.ind_idx, self.args.parents + self.args.children))
        print(f"\33[3mHighscore\33[0m     : \33[1m{self.highscore}\33[0m")
        print(f"\33[3mMean score\33[0m    : \33[1m{round(self.mean_score, 2)}\33[0m")
        print("\33[3mMean fitness\33[0m  : \33[1m{:,}\33[0m".format(round(self.mean_fitness, 2)))
        print(f"\33[90m--------------------------------\33[0m")

    def draw(self):
        self.snake_gui.draw()
        self.network_gui.draw(self.activations, self.individual.genome.params)
        self.info_gui.draw(self.ind_idx, self.generation, self.snake_env.score, self.highscore)

    def keyPressEvent(self, event):
        key_press = event.key()

        if key_press == QtCore.Qt.Key_R:
            self.render = not self.render

        elif key_press == QtCore.Qt.Key_Q:
            self.timer.stop()
            exit(0)

        elif key_press == QtCore.Qt.Key_Up:
            self.fps_idx = min(self.fps_idx + 1, 2)
            self.fps = self.fps_settings[self.fps_idx]
            self.timer.setInterval(1_000 / self.fps)

        elif key_press == QtCore.Qt.Key_Down:
            self.fps_idx = max(self.fps_idx - 1, 0)
            self.fps = self.fps_settings[self.fps_idx]
            self.timer.setInterval(1_000 / self.fps)
            
