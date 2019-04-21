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
        # print(np.tanh(z))
        return np.tanh(z)

    @staticmethod
    def deriv(z):
        # print(z.shape)
        return 1.0-np.tanh(z)**2


class ReLu(object):

    @staticmethod
    def fn(z):
        # print(z.shape) #TODO: DEBUG
        # z[z <= 0] = 0
        # z[z > 0] = z
        z = np.nan_to_num(z)
        z = np.maximum(z, 0)
        return z

    @staticmethod
    def deriv(z):
        # print(z)
        z = np.nan_to_num(z)
        z[z <= 0] = 0
        z[z > 0] = 1
        return z


class Elu(object):

    @staticmethod
    def fn(z):
        # let alpha pos const. = 1
        #TODO DEBUG
        z = np.nan_to_num(z)
        alpha = 1
        for i in range(len(z)):
            if z[i] <= 0:
                z[i] = alpha*(np.exp(z[i])-1)
        return z

    @staticmethod
    def deriv(z):
        z = np.nan_to_num(z)
        alpha = 1
        for i in range(len(z)):
            if z[i] <= 0:
                z[i] = alpha * np.exp(z[i])
            else:
                z[i] = 1
        return z


# class SoftMax(object):
#
#     @staticmethod
#     def fn(z):
#         e_z = np.exp(z - np.max(z))
#         return e_z / e_z.sum(axis=0)
#
#     @staticmethod
#     def deriv(z):
#         soft_max = SoftMax.fn(z)
#         s = soft_max.reshape(-1, 1)
#         q = np.diagflat(s) - np.dot(s, s.T)
#         print(q.shape)
#         return q
