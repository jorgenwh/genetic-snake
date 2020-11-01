import numpy as np 


class Individual:
    def __init__(self, network):
        self.network = network

        self.score = 0
        self.steps = 0
        self.fitness = 0

    
    def compute_fitness(self) -> None:
        # Fitness function from that one paper pepega
        #self.fitness = self.steps * self.steps * 2.0**self.score

        # Improvised fitness function 1
        #self.fitness = max(self.steps * self.steps * 2.0**self.score - self.steps**2.0, .1)

        # Improvised fitness function 2
        self.fitness = self.steps * self.steps * 2.0**self.score - self.steps * self.steps * 2.0**(max(self.score-2, 0))