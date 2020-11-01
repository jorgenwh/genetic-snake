import numpy as np 


settings = {
    # update_window:
    #    Whether or not to continuously render visual updates while running the algorithm. This slows down the training
    #    significantly. 
    #    This setting can be changed at any time by pressing 'S'
    
    # save_best_models:
    #    Whether or not to locally save the current best neural network for the current game size. Each game size will
    #    be stored separately and can be loaded in to play on different sizes than they were trained on later.

    # save_best_game:
    #    Whether or not to save the best game performance to reload and replay it later.

    # load_model:
    #    Whether or not to load a pre-trained model stored from an earlier training session. This will show the loaded model
    #    play the game once.

    # load_game_size:
    #    The game size the neural network you want to load in was trained on. This assumes you have already trained and stored
    #    a neural network for this particular game size already.
    #    When loading a deterministic game, this will choose which game map size is loaded.
    
    'update_window': True,

    'save_best_models': True,
    'save_best_game': True,

    'load_model': False,
    'load_game': False,
    'load_game_size': 10,


    # fps:
    #    Rate of updating for the algorithm. The fps will be slowed down significantly by rendering the widgets while training.
    #    Can be changed to 5, 10, 200 or 10000 at any time by pressing 1, 2, 3 or 4 on your keyboard.

    # draw_grid:
    #    Whether or not to draw the grid of the snake game in the visual rendering. Can be toggled at any point by pressing 'G'.
    
    'fps': 10000,
    'draw_grid': False,


    # cell_size:
    #    Pixel size of each cell of the snake grid in the visual rendering NxN. The snake widgets overall size will depend
    #    on the size of each cell and the amount of cells that makes up the grid map.

    # neuron_radius:
    #    Radius of each neuron drawn in the neural network widget (in pixels).

    # neuron_offset:
    #    Gap factor between each neuron (vertically) in the neural network widget.

    # layer_offset:
    #    Gap factor between each layer of the neural network (horizontally) in the neural network widget.

    'cell_size': 35,
    'neuron_radius': 7.5,
    'neuron_offset': 4.5,
    'layer_offset': 175,


    # map_size:
    #    NxN grid size of the snake game map.

    # max_moves:
    #    How many moves can each snake perform without eating a food before it is terminated.

    # network_structure:
    #    The dimensions of each layer in the neural networks.

    # n_parents:
    #    How many individuals from the population will survive and become potential parents in each generation.

    # n_children:
    #    How many children will be produced each generation.

    # mutation_rate:
    #    How many percent of each child's genome will be mutated.

    # max_generations:
    #    How many generations before the algorithm automatically terminates.

    # crossover_options:
    #    Which (out of the possible) crossover options will have a chance to be used whenever a crossover is performed.
    #    Options are: 'single_point', 'SBX'
    
    # mutation_options:
    #    Which (out of the possible) mutation options will have a chance to be used whenever a mutation is performed.
    #    Options are: 'gaussian'

    # window_title:
    #    Title of the application window.

    'map_size': 10,
    'max_moves': 100,

    'network_structure': [32, 20, 12, 4],
    'n_parents': 500, #250
    'n_children': 1000, # 250, 500
    'mutation_rate': 0.05,
    'max_generations': np.inf,

    'crossover_options': ['single_point', 'SBX'],
    'mutation_options': ['gaussian'],

    'window_title': 'snake-deep-learning-ga'
}