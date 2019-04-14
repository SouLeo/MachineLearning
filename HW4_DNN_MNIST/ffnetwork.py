import numpy as np

class FFNetwork(object):
    def __init__(self, layers_and_dim):
        self.layers_and_dim = layers_and_dim
        self.num_layers = len(self.layers_and_dim)
        self.num_neurons = np.sum(self.layers_and_dim)
        # for MNIST, input must always be 784 and output must always be 10
        if self.layers_and_dim[0] != 784 or self.layers_and_dim[-1] != 10:
            print('WARNING: INPUT AND OUTPUT LAYER DIM. SET INAPPROPRIATELY.')
            print('INPUT ADJUSTED TO 784. OUTPUT ADJUSTED TO 10.')
            self.layers_and_dim[0] = 784
            self.layers_and_dim[-1] = 10
        # randomly generate biases and weights for every neuron
        # self.biases size is layer width by num_layers (starting from first hidden)
        self.biases = [np.random.randn(x, 1) for x in self.layers_and_dim[1:]]
        # self.weights is
        shift = np.roll(self.layers_and_dim, 1)
        weight_combos = np.concatenate((self.layers_and_dim, shift), axis=1)
        weight_combos = np.delete(weight_combos, 0, 0)
        self.weights = [np.random.randn(x, y) for x, y in weight_combos]

