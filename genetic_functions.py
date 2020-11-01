import numpy as np 
import random

from settings import settings

from individual import Individual
from neural_network import NeuralNetwork


def elitist_selection(population: list, n: int) -> list:
    inds = sorted(population, key = lambda individual: individual.fitness, reverse=True)
    return inds[:n]


def roulette_wheel_selection(population: list, n: int):
    selected = []
    wheel = sum(ind.fitness for ind in population)
    for _ in range(n):
        pick = random.uniform(0, wheel)
        current = 0
        for ind in population:
            current += ind.fitness
            if current > pick:
                selected.append(ind)
                break
    
    return selected


def crossover(mom: Individual, dad: Individual) -> tuple:
    m_weights, d_weights = mom.network.weights, dad.network.weights
    son_weights, daughter_weights = [], []

    for m_w, d_w in zip(m_weights, d_weights):
        option = random.choice(settings['crossover_options'])
        if option == 'single_point':
            son, daughter = single_point_crossover(m_w, d_w)
            son_weights.append(son)
            daughter_weights.append(daughter)
        elif option == 'SBX':
            son, daughter = simulated_binary_crossover(m_w, d_w)
            son_weights.append(son)
            daughter_weights.append(daughter)

    assert len(son_weights) == len(m_weights) 

    return (
        Individual(NeuralNetwork(layer_dim=settings['network_structure'], weights=son_weights)),
        Individual(NeuralNetwork(layer_dim=settings['network_structure'], weights=daughter_weights)))


def simulated_binary_crossover(mom: np.ndarray, dad: np.ndarray) -> np.ndarray:
    eta = 100
    rand = np.random.uniform(0, 1, mom.shape)
    gamma = np.empty(mom.shape)

    gamma[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (eta + 1))
    gamma[rand > 0.5] = (1.0 / (2.0 * (1.0 - rand[rand > 0.5]))) ** (1.0 / (eta + 1))

    son = 0.5 * ((1 + gamma) * mom + (1 - gamma) * dad)
    daughter = 0.5 * ((1 - gamma) * mom + (1 + gamma) * dad)

    return son, daughter


def single_point_crossover(mom: np.ndarray, dad: np.ndarray) -> np.ndarray:
    son = mom.copy()
    daughter = dad.copy()

    rows, cols = mom.shape
    row = np.random.randint(0, rows)
    col = np.random.randint(0, cols)

    son[:row, :] = dad[:row, :]
    daughter[:row, :] = mom[:row, :]

    son[row, :col+1] = dad[row, :col+1]
    daughter[row, :col+1] = mom[row, :col+1]

    return son, daughter


def mutate(individual: Individual) -> None:
    for w in individual.network.weights:
        gaussian_mutation(w)


def gaussian_mutation(genome: np.ndarray) -> np.ndarray:
    scale = .2
    mutation_rate = settings['mutation_rate']

    mutation_array = np.random.uniform(0, 1, size=genome.shape) < mutation_rate
    gaussian_mutation = np.random.normal(size=genome.shape)

    gaussian_mutation[mutation_array] *= scale

    genome[mutation_array] += gaussian_mutation[mutation_array]