from PyQt5 import QtGui, QtWidgets
import numpy as np

class InfoGui(QtWidgets.QWidget):
    def __init__(self, parent, args):
        super().__init__(parent)
        self.args = args
        self.population_size = self.args.nparents + self.args.nchildren
        self.ind_idx = 0
        self.generation = 0
        self.score = 0
        self.highscore = 0
        self.show()

    def draw(self, ind_idx: int, generation: int, score: int, highscore: int):
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
        painter.drawText(middle, 25, "{:,} / {:,}".format(self.ind_idx, self.population_size))

        painter.drawText(0, 50, "generation :")
        painter.drawText(middle, 50, "{:,}".format(self.generation))

        painter.drawText(0, 75, "score :")
        painter.drawText(middle, 75, f"{self.score}")

        painter.drawText(0, 100, "highest score :")
        painter.drawText(middle, 100, f"{self.highscore}")

        # commands
        painter.drawText(0, 150, "toggle rendering :")
        painter.drawText(middle, 150, "R")

        painter.drawText(0, 175, "change FPS / speed :")
        painter.drawText(middle, 175, "Arrow up/down")

        painter.drawText(0, 200, "quit:")
        painter.drawText(middle, 200, "Q")