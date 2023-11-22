import os
import numpy as np
from typing import List

class Genome():
    """
    Class representing the genome of a individual solution. The genome is made up
    of the neural network parameters.
    """
    def __init__(self, dims: List[int] = [32, 20, 12, 4], params: List[np.ndarray] = []):
        self.dims = dims
        self.params = params
        
        # If no parameters are provided, we initialize new random parameters
        if not self.params:
            for i in range(len(self.dims) - 1):
                w_m = np.random.uniform(-1, 1, size=(self.dims[i], self.dims[i+1]))
                self.params.append(w_m)

    # Forward x through the network and return the activations at each layer 
    def forward(self, x: np.ndarray) -> List[np.ndarray]:
        # Store each layer's activation
        activations = [x]

        for i in range(len(self.params)):
            zh = np.dot(activations[-1], self.params[i])

            # ReLU activation for all hidden layers and linear activation for output layer
            if i != len(self.params) - 1:
                ha = self.relu(zh)
            else:
                ha = zh

            # store activation
            activations.append(ha)
        
        return activations

    def relu(self, x):
        return np.maximum(x, 0)

    def save(self, dir_name):
        if os.path.isdir(dir_name):
            print(f"Error: cannot save genome because directory '{dir_name}' already exists.")
            exit(1)

        os.mkdir(dir_name)
        for i, p in enumerate(self.params):
            filename = os.path.join(dir_name, f"w{i+1}.npy")
            np.save(filename, p)

    def load(self, dir_name):
        self.dims = []
        self.params = []

        p_filenames = sorted(os.listdir(dir_name))
        for i, filename in enumerate(p_filenames):
            p = np.load(os.path.join(dir_name, filename))
            print(f"Loaded parameter matrix {i+1} with shape {p.shape}")
            self.params.append(p)

            self.dims.append(p.shape[0])
            if i == len(p_filenames) - 1:
                self.dims.append(p.shape[1])
