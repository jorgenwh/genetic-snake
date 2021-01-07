from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np

from .agent import Agent
from snake_env import Snake_Env

from snake_widget import Snake_Widget
from nn_widget import NN_Widget
from text_widget import Text_Widget

class Window(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.agent = Agent(self.args)
        self.snake_env = Snake_Env(self.args, version=1)
        self.observation = self.snake_env.reset()

        self.episode = 0
        self.highscore = 0
        self.action = 0

        self.visualize = True
        self.fps_idx = 2
        self.fps_settings = [3, 10, 10_000]
        self.fps = self.fps_settings[self.fps_idx]

        self.init_window()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(1_000 / self.fps)
        self.show()

    def init_window(self):
        snake_widget_size = self.args.size * 35, self.args.size * 35
        nn_widget_size = (4 * 175) - 40, 32 * ((7.5 * 2) + 4.5)
        width, height = 85 + snake_widget_size[0] + nn_widget_size[0], 80 + max(nn_widget_size[1], snake_widget_size[1])

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

    def step(self):
        if self.snake_env.terminal:
            self.episode += 1
            self.observation = self.snake_env.reset()
        else:
            self.action = self.agent.act(self.observation)
            observation, reward = self.snake_env.step(self.action)
            self.highscore = max(self.highscore, self.snake_env.score)

            self.agent.remember(self.observation, self.action, reward, observation, self.snake_env.terminal)
            self.agent.learn()

            self.observation = observation

        #print(f"episode: {self.episode + 1} - loss: {self.agent.loss_meter} - highscore: {self.highscore} - epsilon: {round(self.agent.epsilon, 3)}   \r", end="")
        if self.visualize:
            self.draw()

    def draw(self):
        params = []
        layer_dims = [32, 20, 12, 4]

        for i in range(len(layer_dims) - 1):
            w = np.full((layer_dims[i], layer_dims[i+1]), fill_value=None)
            params.append(w)
        
        activations = [np.zeros((1, l_size)) for l_size in layer_dims]
        activations[-1][0,self.action] = 1

        self.snake_widget.draw()
        self.nn_widget.draw(activations, params)

    def keyPressEvent(self, event):
        key_press = event.key()

        if key_press == QtCore.Qt.Key_S:
            self.visualize = not self.visualize

        elif key_press == QtCore.Qt.Key_E:
            self.timer.stop()
            exit()

        elif key_press == QtCore.Qt.Key_Up:
            self.fps_idx = min(self.fps_idx + 1, 2)
            self.fps = self.fps_settings[self.fps_idx]
            self.timer.setInterval(1_000 / self.fps)

        elif key_press == QtCore.Qt.Key_Down:
            self.fps_idx = max(self.fps_idx - 1, 0)
            self.fps = self.fps_settings[self.fps_idx]
            self.timer.setInterval(1_000 / self.fps)
            