import numpy as np 

class Individual:
    def __init__(self, nnet):
        self.nnet = nnet
        self.fitness = 0

    def act(self, observation):
        return self.nnet.forward(observation)

    def compute_fitness(self, score, steps):
        self.fitness = steps * steps * 2.0 ** score - steps * steps * 2.0 ** (max(score - 2, 0))