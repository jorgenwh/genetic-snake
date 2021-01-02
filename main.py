import sys
import argparse

from PyQt5 import QtGui, QtCore, QtWidgets
from genetic_algorithm.window import Window

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake game.")

    parser.add_argument("--size", help="Snake game grid size", type=int, default=10)
    parser.add_argument("--parents", help="Number of parents to choose for selection per generation.", type=int, default=500)
    parser.add_argument("--children", help="Number of children to produce per generation.", type=int, default=1000)
    parser.add_argument("--step_limit", help="Number of steps a snake can do without eating a food before dying.", type=int, default=100)

    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = Window(args)
    sys.exit(app.exec_())