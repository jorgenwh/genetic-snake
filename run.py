import sys
import argparse
from PyQt5 import QtWidgets

from genetic_snake import GUIApplication
from genetic_snake import SnakeEnv
from genetic_snake import run_genetic_algorithm
from genetic_snake import Individual
from genetic_snake import Genome 
import genetic_snake_C

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Genetic snake")
    parser.add_argument("-size", help="The grid size of the snake environment.", type=int, default=10)
    parser.add_argument("-parents", help="The number of surviving individuals that will be selected for reproduction each generation.", type=int, default=750)
    parser.add_argument("-children", help="The number of children to produce through reproduction each generation.", type=int, default=750)
    parser.add_argument("-ind", help="The path to a directory storing a neural network checkpoint. When provided, the individual is loaded and plays one game of snake displayed using the GUI.", type=str, default=None)
    parser.add_argument("-threads", help="Number of threads to use to parallelize the game-play portion of the genetic algorithm. Parallel processing is only enabled when running with the --nogui flag.", type=int, default=1)
    parser.add_argument("-name", help="Name of the output directory storing the neural network parameters of the first snake to beat the game.", type=str, default=None)
    parser.add_argument("--usecpp", action="store_true", help="Use C++ for greater performance.")
    parser.add_argument("--nogui", action="store_true", help="Run the genetic algorithm without a GUI. This results in significantly faster training and enables multithreading.")
    args = parser.parse_args()
    print("ARGS:", args)

    if args.usecpp:
        SnakeEnv = genetic_snake_C.SnakeEnv

    loaded_ind = None
    if args.ind is not None:
        print(f"Loading individual: '{args.ind}'")
        genome = Genome()
        genome.load(args.ind)
        loaded_ind = Individual(genome)

    if args.nogui and args.ind is None:
        run_genetic_algorithm(args.parents, args.children, args.usecpp, args.size, SnakeEnv, args.threads, output_dir_name=args.name) 
    else:
        application = QtWidgets.QApplication(sys.argv)
        gui = GUIApplication(args, snake_env_class=SnakeEnv, loaded_individual=loaded_ind, output_dir_name=args.name)
        sys.exit(application.exec_())
