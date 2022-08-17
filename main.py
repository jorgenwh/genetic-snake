import sys
import argparse
from PyQt5 import QtWidgets

from genetic_snake.gui.gui_application import GuiApplication
from genetic_snake.genetic.ga import run_genetic_algorithm
from genetic_snake.snake_env import SnakeEnv
from genetic_snake.genetic.individual import Individual
from genetic_snake.genetic.genome import Genome 

# Run build script to build the cpp module. Requires CXX and pybind11
import genetic_snake.cpp_module as cpp_module

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic snake game player.")

    # Snake
    parser.add_argument("-size", help="Snake game grid size", type=int, default=10)
    
    # Genetic algorithm
    parser.add_argument("-parents", help="Number of parents to choose for selection per generation.", type=int, default=750)
    parser.add_argument("-children", help="Number of children to produce per generation.", type=int, default=750)
    parser.add_argument("-ind", help="Name of an individual to load and play.", type=str, default=None)
    parser.add_argument("-threads", help="Number of threads used for simulating evaluation games when cpp module is used.", type=int, default=1)
    parser.add_argument("--usecpp", action="store_true")
    parser.add_argument("--nogui", action="store_true")

    args = parser.parse_args()

    if args.usecpp:
        SnakeEnv = cpp_module.SnakeEnv

    loaded_ind = None
    if args.ind is not None:
        print(f"Loading individual: '{args.ind}'")
        genome = Genome()
        genome.load(args.ind)
        loaded_ind = Individual(genome)

    if args.nogui:
        run_genetic_algorithm(args.parents, args.children, args.usecpp, args.size, SnakeEnv, args.threads) 
    else:
        application = QtWidgets.QApplication(sys.argv)
        gui = GuiApplication(args, snake_env_class=SnakeEnv, loaded_individual=loaded_ind)
        sys.exit(application.exec_())
