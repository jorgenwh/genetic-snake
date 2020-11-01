import numpy as np 
import random 

class NeuralNetwork:
    def __init__(self, layer_dim=[32, 20, 12, 4], weights=None):
        self.layer_dim = layer_dim
        self.weights = weights

        # Initialize the weights
        if not weights:
            self.weights = []
            #sigma = np.power(self.layer_dim[0], -0.5)
            for i in range(len(self.layer_dim) - 1):
                #w = sigma * np.random.randn(self.layer_dim[i], self.layer_dim[i+1]) + 0
                w = np.random.uniform(-1, 1, size=(self.layer_dim[i], self.layer_dim[i+1]))
                self.weights.append(w)

        self.input_vector = None
        self.activations = []

    def forward(self, input_vector):
    # Feed the input vector through the network and store the activations
        self.activations.clear()
        self.input_vector = input_vector

        for i in range(len(self.weights)):
            
            # Forward the input through each layer
            if i == 0:
                zh = input_vector @ self.weights[i]
            else:
                zh = self.activations[-1] @ self.weights[i]

            # ReLU activation for all hidden layers and linear activation for output layer
            if i != len(self.weights) - 1:
                ha = self.relu(zh)
            else:
                ha = zh

            # Store each layer's activations
            self.activations.append(ha)

    def relu(self, x):
    # ReLU activation function
        return np.maximum(x, 0)