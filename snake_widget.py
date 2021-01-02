from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np

class Snake_Widget(QtWidgets.QWidget):
    def __init__(self, parent, snake_env, args):
        super().__init__(parent)
        self.args = args
        self.snake_env = snake_env
        self.show()

    def draw(self):
        self.repaint()

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)

        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtCore.Qt.black))

        self.draw_map(painter)
        self.draw_snake(painter)
        self.draw_food(painter)
        painter.end()

    def draw_map(self, painter):
        width = self.frameGeometry().width()
        height = self.frameGeometry().height()

        painter.setPen(QtCore.Qt.black)

        for i in range(self.args.size + 1):
            y = i * 35
            x = i * 35

            if i == 0 or i == self.args.size:
                painter.drawLine(0, y, width, y)
                painter.drawLine(x, 0, x, height)

    def draw_snake(self, painter):
        pen = QtGui.QPen()
        pen.setColor(QtGui.QColor(0, 0, 0))
        painter.setPen(pen)
        brush = QtGui.QBrush()
        brush.setColor(QtCore.Qt.green)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(20, 165, 10)))

        for x, y in self.snake_env.snake_body:
            if (x, y) == self.snake_env.snake_body[0]:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 0)))
            else:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(20, 165, 10)))
            painter.drawRect(
                x * 35, y * 35, 35, 35)

            if (x, y) == self.snake_env.snake_body[0]:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
                eye_radius = 35 / 9.0

                # draw eyes
                painter.drawEllipse(
                    x * 35 + (35 / 4.0), 
                    y * 35 + (35 / 3.5), 
                    35 / 9.0, 
                    35 / 9.0
                )

                painter.drawEllipse(
                    x * 35 + ((3.0 / 4.0) * 35) - eye_radius / 2.0,
                    y * 35 + (35 / 3.5), 
                    35 / 9.0, 
                    35 / 9.0
                )
                
                # draw mouth
                painter.drawLine(
                    x * 35 + (35 / 4.0) + eye_radius / 2.0, 
                    y * 35 + (35 / 3.5) * 2.5, 
                    x * 35 + ((3.0 / 4.0) * 35) - eye_radius / 2.0,
                    y * 35 + (35 / 3.5) * 2.5
                )

    def draw_food(self, painter):
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setPen(QtGui.QPen(QtCore.Qt.black))
        painter.setBrush(QtGui.QBrush(QtCore.Qt.red))

        x, y = self.snake_env.food
        painter.drawRect(x * 35, y * 35, 35, 35)