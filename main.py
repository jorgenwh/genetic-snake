import sys
import argparse

from PyQt5 import QtGui, QtCore, QtWidgets
from genetic_algorithm.program import Window as GA
from deep_q_learning.program import Window as DQN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake game.")

    # Snake
    parser.add_argument("--method", help="Which learning method to use to train the neural network.", choices=["genetic-algorithm", "deep-q-learning"], default="genetic-algorithm")
    parser.add_argument("--size", help="Snake game grid size", type=int, default=10)
    
    # Genetic algorithm
    parser.add_argument("--parents", help="Number of parents to choose for selection per generation.", type=int, default=500)
    parser.add_argument("--children", help="Number of children to produce per generation.", type=int, default=1000)
    parser.add_argument("--step_limit", help="Number of steps a snake can do without eating a food before dying.", type=int, default=100)

    # Deep Q learning
    parser.add_argument("--cuda", help="Enable cuda.", type=bool, default=True)
    parser.add_argument("--lr", help="Neural network learning rate.", type=float, default=1e-3)
    parser.add_argument("--batch_size", help="Neural network training batch size.", type=int, default=256)
    parser.add_argument("--epsilon", help="Starting epsilon value.", type=float, default=1.0)
    parser.add_argument("--gamma", help="Transition discount factor.", type=float, default=0.97)
    parser.add_argument("--replay_memory", help="Maximum number of transitions to store.", type=int, default=100_000)

    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    if args.method == "genetic-algorithm":
        window = GA(args)
    elif args.method == "deep-q-learning":
        window = DQN(args)

    sys.exit(app.exec_())