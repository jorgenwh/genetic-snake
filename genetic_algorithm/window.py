from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np

from .individual import Individual
from .nnet import Neural_Network
from snake_env import Snake_Env
from .genetic_functions import elitist_selection, roulette_wheel_selection, crossover, mutate

from snake_widget import Snake_Widget
from nn_widget import NN_Widget
from text_widget import Text_Widget

class Window(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.population = [Individual(Neural_Network(layer_dims=[32, 20, 12, 4])) for _ in range(self.args.parents + self.args.children)]
        self.ind_idx = 0
        self.individual = self.population[self.ind_idx]
        self.snake_env = Snake_Env(self.args)
        self.observation = self.snake_env.reset()

        self.generation = 1
        self.highscore = 0
        self.mean_score = 0
        self.mean_fitness = 0
        self.activations = []
        self.generation_score = 0

        self.init_window()
        self.fps = 10_000
        self.visualize = True
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(1_000 / self.fps)
        self.show()

    def init_window(self):
        snake_widget_size = self.args.size * 35, self.args.size * 35
        nn_widget_size = (4 * 175) - 40, 32 * ((7.5 * 2) + 4.5)
        text_widget_size = snake_widget_size[0], 255
        width, height = 85 + snake_widget_size[0] + nn_widget_size[0], 80 + max(nn_widget_size[1], snake_widget_size[1] + text_widget_size[1], text_widget_size[1])

        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setGeometry(800, 300, width, height)
        self.setFixedSize(width, height)

        # snake game widget
        self.snake_widget = Snake_Widget(self.centralWidget, self.snake_env, self.args)
        self.snake_widget.setGeometry(35, 50, snake_widget_size[0], snake_widget_size[1])

        # neural network widget
        self.nn_widget = NN_Widget(self.centralWidget, [32, 20, 12, 4])
        self.nn_widget.setGeometry(110 + snake_widget_size[0], 50, nn_widget_size[0], nn_widget_size[1])

        # text widget
        self.text_widget = Text_Widget(self.centralWidget, self.args)
        self.text_widget.setGeometry(35, 50 + snake_widget_size[0], text_widget_size[0], text_widget_size[1])

    def step(self):
        if self.snake_env.terminal:
            self.ind_idx += 1
            self.individual.compute_fitness(self.snake_env.score, self.snake_env.steps)
            self.generation_score += self.snake_env.score

            if self.ind_idx == self.args.parents + self.args.children:
                self.generation_step()
                self.generation += 1
                self.ind_idx = 0
                self.individual = self.population[self.ind_idx]
            else:
                self.individual = self.population[self.ind_idx]
                self.observation = self.snake_env.reset()

        else:
            self.activations = self.individual.act(self.observation)
            action = np.argmax(self.activations[-1])
            self.observation, _ = self.snake_env.step(action)

            self.highscore = max(self.highscore, self.snake_env.score)

        if self.visualize:
            self.draw()

        print(f"individual: {self.ind_idx + 1} - generation: {self.generation} - score: {self.snake_env.score} - highscore: {self.highscore} - mean_score: {round(self.mean_score, 1)}     \r", end="")

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

    def draw(self):
        self.snake_widget.draw()
        self.nn_widget.draw(self.activations, self.individual.nnet)
        self.text_widget.draw(self.ind_idx, self.generation, self.snake_env.score, self.highscore)

    def keyPressEvent(self, event):
        key_press = event.key()

        if key_press == QtCore.Qt.Key_S:
            self.visualize = not self.visualize

        elif key_press == QtCore.Qt.Key_E:
            self.timer.stop()
            exit()