from .genome import Genome

class Individual():
    """
    Class representing an individual solution, storing the solution's fitness,
    the genome (network) and implements some necessary methods.
    """
    def __init__(self, genome: Genome = None):
        self.genome = genome

        # If no genome was provided, create a random genome
        if self.genome is None:
            self.genome = Genome()

        self.fitness = 0

    def act(self, observation):
        return self.genome.forward(observation)

    def compute_fitness(self, score, steps):
        self.fitness = (steps * 2.0 ** score) - (steps * 2.0 ** (max(score - 2.25, 0))) + (score ** 3.5)
