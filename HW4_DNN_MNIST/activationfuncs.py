import numpy as np


class Sigmoid(object):

    @staticmethod
    def fn(z):
        ans = 1.0 / (1.0 + np.exp(-z))
        return ans

    @staticmethod
    def deriv(z):
        return Sigmoid.fn(z)*(1-Sigmoid.fn(z))

# def sigmoid(z):
#     ans = 1.0/(1.0 + np.exp(-z))
#     return ans
#
#
# def sigmoid_deriv(z):
#     return sigmoid(z)*(1-sigmoid(z))
#
# # TODO: Add tanh, ReLu, and ELU activation functions
