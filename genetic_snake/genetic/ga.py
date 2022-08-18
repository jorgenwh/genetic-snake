import numpy as np

from genetic_snake.genetic.genome import Genome
from genetic_snake.genetic.individual import Individual 
from genetic_snake.genetic.functional import elitist_selection, roulette_wheel_selection
from genetic_snake.genetic.functional import crossover 
from genetic_snake.genetic.functional import mutate

import genetic_snake.cpp_module as cpp_module

def evaluate_population(population, size, snake_env_class):
    highscore = 0
    mean_score = 0
    mean_fitness = 0
    population_size = len(population)
    snake_env = snake_env_class(size)

    for i, ind in enumerate(population):
        print("\33[3mIndividual\33[0m    : \33[1m{:,}\33[0m / \33[1m{:,}\33[0m".format(i, population_size), end="\r")

        observation = snake_env.reset()

        while not snake_env.is_terminal():
            activations = ind.act(observation)
            action = np.argmax(activations[-1])
            observation = snake_env.step(action)

        score = snake_env.score
        if score == size**2 - 3:
            ind.genome.save("models/ind")

        ind.compute_fitness(score, snake_env.steps)
        highscore = max(highscore, score)

        mean_score += score
        mean_fitness += ind.fitness

    print("\33[3mIndividual\33[0m    : \33[1m{:,}\33[0m / \33[1m{:,}\33[0m".format(population_size, population_size))

    mean_score /= population_size
    mean_fitness /= population_size

    return highscore, mean_score, mean_fitness

def generation_step(population, num_parents, num_children):
    population = elitist_selection(population, num_parents)
    children = []
    while len(children) < num_children:
        mom, dad = roulette_wheel_selection(population, 2)

        son, daughter = crossover(mom, dad)

        mutate(son)
        mutate(daughter)

        children.append(son)
        children.append(daughter)

    population.extend(children)
    return population

def run_genetic_algorithm(parents, children, usecpp, size, snake_env_class, num_threads):
    highscore = 0
    population  = [Individual() for _ in range(parents + children)]
    generation = 1
    pop_size = parents + children

    while True:
        print("\n\33[90m--------------------------------\33[0m\n\33[92mGeneration {:,}\33[0m".format(generation))
        if usecpp:
            game_results = cpp_module.evaluate_population([ind.genome.params for ind in population], size, num_threads)
            game_steps = [result[0] for result in game_results]
            game_scores = [result[1] for result in game_results]
            mean_score = 0
            mean_fitness = 0
            gen_highscore = 0

            for steps, score, ind in zip(game_steps, game_scores, population):
                ind.compute_fitness(score, steps)
                gen_highscore = max(gen_highscore, score)
                mean_score += score
                mean_fitness += ind.fitness

                if score >= size**2 - 3:
                    ind.genome.save(f"models/ind")
                    exit(0)
            
            mean_score /= pop_size
            mean_fitness /= pop_size

        else:
            gen_highscore, mean_score, mean_fitness = evaluate_population(population, size, snake_env_class)
        highscore = max(highscore, gen_highscore)

        print(f"\33[3mHighscore\33[0m     : \33[1m{highscore}\33[0m")
        print(f"\33[3mMean score\33[0m    : \33[1m{round(mean_score, 2)}\33[0m")
        print("\33[3mMean fitness\33[0m  : \33[1m{:,}\33[0m".format(round(mean_fitness, 2)))
        print(f"\33[90m--------------------------------\33[0m")

        population = generation_step(population, parents, children)
        generation += 1

