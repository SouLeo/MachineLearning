import numpy as np


class CrossEntropyLoss(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(active_func, z, a, y):
        return a-y


class QuadLoss(object):

    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(active_func, z, a, y):
        return (a-y) * active_func.fn(z)

