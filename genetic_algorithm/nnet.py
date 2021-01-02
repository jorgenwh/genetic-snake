import numpy as np

class Neural_Network:
    def __init__(self, layer_dims=[32, 20, 12, 4], params=None):
        self.layer_dims = layer_dims
        self.params = params
        
        if not self.params:
            # initialize the params
            self.params = []
            #sigma = np.power(self.layer_dim[0], -0.5)
            for i in range(len(self.layer_dims) - 1):
                #w = sigma * np.random.randn(self.layer_dim[i], self.layer_dim[i+1]) + 0
                w = np.random.uniform(-1, 1, size=(self.layer_dims[i], self.layer_dims[i+1]))
                self.params.append(w)

    def forward(self, observation):
        activations = [observation]

        for i in range(len(self.params)):
            zh = np.dot(activations[-1], self.params[i])

            # ReLU activation for all hidden layers and linear activation for output layer
            if i != len(self.params) - 1:
                ha = self.relu(zh)
            else:
                ha = zh

            # store each layer's activations
            activations.append(ha)
        
        return activations

    def relu(self, x):
        return np.maximum(x, 0)