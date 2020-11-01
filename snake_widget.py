from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np
import random

from settings import settings
from miscellaneous import Point
from snake import Snake


class SnakeWidget(QtWidgets.QWidget):
    def __init__(self, parent, map_size=10, cell_size=30, draw_grid=False, snake=None):
        super().__init__(parent)

        self.map_size = map_size
        self.cell_size = cell_size
        self.draw_grid = draw_grid
        self.snake = snake

        self.show()

    
    def update(self, update_window: bool, draw_grid: bool) -> None:
        self.draw_grid = draw_grid
        if self.snake.is_alive:
            self.snake.update()
            if update_window:
                self.repaint()


    def new_game(self, snake: Snake) -> None:
        self.snake = snake


    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)

        self.draw_map(painter)
        self.draw_snake(painter)
        if not self.snake.has_won:
            self.draw_food(painter)

        painter.end()


    def draw_map(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QtGui.QPen(QtCore.Qt.black))

        width = self.frameGeometry().width()
        height = self.frameGeometry().height()

        painter.setPen(QtCore.Qt.black)

        for i in range(self.map_size + 1):
            y = i * self.cell_size
            x = i * self.cell_size

            if i == 0 or i == self.map_size:
                painter.drawLine(0, y, width, y)
                painter.drawLine(x, 0, x, height)
            else:
                if self.draw_grid:
                    painter.drawLine(0, y, width, y)
                    painter.drawLine(x, 0, x, height)


    def draw_snake(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        pen = QtGui.QPen()
        pen.setColor(QtGui.QColor(0, 0, 0))
        painter.setPen(pen)
        brush = QtGui.QBrush()
        brush.setColor(QtCore.Qt.green)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(20, 165, 10)))

        for x, y in self.snake.snake_body:
            if (x, y) == self.snake.snake_body[0]:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 0)))
            else:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(20, 165, 10)))
            painter.drawRect(
                x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)

            if (x, y) == self.snake.snake_body[0]:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
                eye_radius = self.cell_size / 9.0

                # Draw eyes
                painter.drawEllipse(
                    x * self.cell_size + (self.cell_size / 4.0), 
                    y * self.cell_size + (self.cell_size / 3.5), 
                    self.cell_size / 9.0, 
                    self.cell_size / 9.0)

                painter.drawEllipse(
                    x * self.cell_size + ((3.0 / 4.0) * self.cell_size) - eye_radius / 2.0,
                    y * self.cell_size + (self.cell_size / 3.5), 
                    self.cell_size / 9.0, 
                    self.cell_size / 9.0)

                # Draw mouth
                painter.drawLine(
                    x * self.cell_size + (self.cell_size / 4.0) + eye_radius / 2.0, 
                    y * self.cell_size + (self.cell_size / 3.5) * 2.5, 
                    x * self.cell_size + ((3.0 / 4.0) * self.cell_size) - eye_radius / 2.0,
                    y * self.cell_size + (self.cell_size / 3.5) * 2.5
                )

            
    def draw_food(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setPen(QtGui.QPen(QtCore.Qt.black))
        painter.setBrush(QtGui.QBrush(QtCore.Qt.red))

        x, y = self.snake.food_pos
        painter.drawRect(
            x * self.cell_size, y * self.cell_size, 
            self.cell_size, self.cell_size)