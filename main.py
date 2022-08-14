import sys
import argparse
from PyQt5 import QtWidgets

from genetic_snake.gui.gui_application import GuiApplication
from genetic_snake.snake_env import SnakeEnv

# Run build script to build the cpp module. Requires CXX and pybind11
import genetic_snake.cpp_module as cpp_module

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic snake game player.")

    # Snake
    parser.add_argument("--size", help="Snake game grid size", type=int, default=10)
    
    # Genetic algorithm
    parser.add_argument("--nparents", help="Number of parents to choose for selection per generation.", type=int, default=500)
    parser.add_argument("--nchildren", help="Number of children to produce per generation.", type=int, default=1000)
    parser.add_argument("--usecpp", action="store_true")

    args = parser.parse_args()

    if args.usecpp:
        SnakeEnv = cpp_module.SnakeEnv

    application = QtWidgets.QApplication(sys.argv)
    gui = GuiApplication(args, snake_env_class=SnakeEnv)
    sys.exit(application.exec_())
