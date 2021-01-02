from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np

class Text_Widget(QtWidgets.QWidget):
    def __init__(self, parent, args):
        super().__init__(parent)
        self.args = args
        self.population_size = self.args.parents + self.args.children

        self.score = 0
        self.highest_score = 0

        self.show()

    def draw(self, ind_idx, generation, score, highscore):
        self.ind_idx = ind_idx
        self.generation = generation
        self.score = score
        self.highscore = highscore

        self.repaint()

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        self.draw_stats(painter)
        painter.end()

    def draw_stats(self, painter):
        middle = self.frameGeometry().width() / 2.0

        # pepehands
        painter.drawText(0, 25, "individual :")
        painter.drawText(middle, 25, f"{self.ind_idx + 1} / {self.population_size}")

        painter.drawText(0, 50, "generation :")
        painter.drawText(middle, 50, f"{self.generation}")

        painter.drawText(0, 75, "score :")
        painter.drawText(middle, 75, f"{self.score}")

        painter.drawText(0, 100, "highest score :")
        painter.drawText(middle, 100, f"{self.highscore}")

        # commands
        painter.drawText(0, 150, "toggle visual update :")
        painter.drawText(middle, 150, "S")

        painter.drawText(0, 175, "change FPS / speed :")
        painter.drawText(middle, 175, "Arrow key up / down")

        painter.drawText(0, 200, "terminate program :")
        painter.drawText(middle, 200, "E")