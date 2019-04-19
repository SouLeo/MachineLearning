import numpy as np


class Sigmoid(object):

    @staticmethod
    def fn(z):
        ans = 1.0 / (1.0 + np.exp(-z))
        return ans

    @staticmethod
    def deriv(z):
        return Sigmoid.fn(z)*(1-Sigmoid.fn(z))

class Tanh(object):

    @staticmethod
    def fn(z):
        return np.tanh(z)

    @staticmethod
    def deriv(z):
        1.0-np.tanh(z)**2

class ReLu(object):

    @staticmethod
    def fn(z):
        z[z <= 0] = 0
        # z[z > 0] = z
        return z

    @staticmethod
    def deriv(z):
        z[z <= 0] = 0
        z[z > 0] = 1
        return z

class Elu(object):

    @staticmethod
    def fn(z):
        # let alpha pos const. = 1
        alpha = 1
        z[z <= 0] = alpha*(np.exp(z)-1)

    @staticmethod
    def deriv(z):
        alpha = 1
        z[z > 0] = 1
        z[z < 0] = alpha*np.exp(z)

