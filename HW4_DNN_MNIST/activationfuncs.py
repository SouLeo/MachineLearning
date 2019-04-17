import numpy as np


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))

# TODO: Add tanh, ReLu, and ELU activation functions
