from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np
import random

from miscellaneous import Point


class TextWidget(QtWidgets.QWidget):
    def __init__(self, parent, population_size):
        super().__init__(parent)

        self.cur_ind = 0
        self.generation = 0
        self.population_size = population_size

        self.score = 0
        self.highest_score = 0
        self.average_fitness = 0

        self.show()

    
    def update(self, update_window: bool, cur_ind: int, generation: int, score: int, highest_score: int, average_fitness: float) -> None:
        self.cur_ind = cur_ind
        self.generation = generation
        self.score = score
        self.highest_score = highest_score
        self.average_fitness = average_fitness

        if update_window:
            self.repaint()

    
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)

        self.draw_stats(painter)

        painter.end()

    
    def draw_stats(self, painter: QtGui.QPainter) -> None:
        middle = self.frameGeometry().width() / 2.0

        # pepehands
        painter.drawText(0, 25, "individual :")
        painter.drawText(middle, 25, f"{self.cur_ind} / {self.population_size}")

        painter.drawText(0, 50, "generation :")
        painter.drawText(middle, 50, f"{self.generation}")

        painter.drawText(0, 75, "score :")
        painter.drawText(middle, 75, f"{self.score}")

        painter.drawText(0, 100, "highest score :")
        painter.drawText(middle, 100, f"{self.highest_score}")

        painter.drawText(0, 125, "average fitness :")
        painter.drawText(middle, 125, f"{round(self.average_fitness, 2)}")

        # Commands
        painter.drawText(0, 175, "toggle visual update :")
        painter.drawText(middle, 175, "S")

        painter.drawText(0, 200, "toggle grid :")
        painter.drawText(middle, 200, "G")

        painter.drawText(0, 225, "change FPS / speed :")
        painter.drawText(middle, 225, "1 - 4")

        painter.drawText(0, 250, "terminate program :")
        painter.drawText(middle, 250, "E")