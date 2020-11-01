import os
import numpy as np 


def save_model(weights: list, layer_dim: list, map_size: int) -> None:
    # Create a directory for the model if one doesn't already exist
    if not os.path.isdir('models'):
        os.mkdir('models')

    if not os.path.isdir(f'models/model_{map_size}'):
        os.mkdir(f'models/model_{map_size}')
        os.mkdir(f'models/model_{map_size}/weights')

    # Save the network layer dimensions
    layer_dim_np = np.array(layer_dim)
    np.save(f'models/model_{map_size}/layer_dim.npy', layer_dim_np)

    # Save each of the weights
    for i, w in enumerate(weights):
        np.save(f'models/model_{map_size}/weights/w{i}.npy', w)


def load_model(map_size: int) -> tuple:
    if not os.path.isdir(f'models/model_{map_size}'):
        print("Failed to load model: model data for this map size doesn't exist")
        quit(1)

    layer_dim = np.load(f'models/model_{map_size}/layer_dim.npy').tolist()

    # Load the network
    weights = []
    for i in range(len(layer_dim) - 1):
        w = np.load(f'models/model_{map_size}/weights/w{i}.npy')
        weights.append(w)

    return layer_dim, weights


def save_game(game_data: dict, map_size: int) -> None:
    # Make all necessary folders if they don't exist
    if not os.path.isdir('game_data'):
        os.mkdir('game_data')

    if not os.path.isdir(f'game_data/game_{map_size}'):
        os.mkdir(f'game_data/game_{map_size}')

    # Save the starting position
    start_pos = game_data['start_position']

    f = open(f'game_data/game_{map_size}/start_pos.txt', 'w')
    f.write(f'{start_pos[0]} {start_pos[1]}')
    f.close()

    # Save the action list
    actions = np.array(game_data['directions'])
    np.save(f'game_data/game_{map_size}/directions.npy', actions)

    # Save the food spawn positions
    food_spawns = np.array(game_data['food_spawns'])
    np.save(f'game_data/game_{map_size}/food_spawns.npy', food_spawns)


def load_game(map_size) -> dict:
    if not os.path.isdir(f'game_data/game_{map_size}'):
        print("Failed to load game: game data for this map size doesn't exist")
        quit(1)

    directions = np.load(f'game_data/game_{map_size}/directions.npy').tolist()
    food_spawns = np.load(f'game_data/game_{map_size}/food_spawns.npy').tolist()

    f = open(f'game_data/game_{map_size}/start_pos.txt', 'r')
    read = f.read()
    read_list = read.split()
    start_pos = (int(read_list[0]), int(read_list[1]))

    game_data = {'directions': directions, 'food_spawns': food_spawns, 'start_position': start_pos}

    return game_data