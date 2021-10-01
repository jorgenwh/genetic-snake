import sys
import argparse
from PyQt5 import QtWidgets

from genetic_snake.gui.gui_application import GuiApplication

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic snake game player.")

    # Snake
    parser.add_argument("--size", help="Snake game grid size", type=int, default=10)
    
    # Genetic algorithm
    parser.add_argument("--nparents", help="Number of parents to choose for selection per generation.", type=int, default=500)
    parser.add_argument("--nchildren", help="Number of children to produce per generation.", type=int, default=1000)

    args = parser.parse_args()

    application = QtWidgets.QApplication(sys.argv)
    gui = GuiApplication(args) 
    sys.exit(application.exec_())
