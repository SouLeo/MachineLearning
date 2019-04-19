import numpy as np


class CrossEntropyLoss(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(a, y):
        return a-y


class HingeLoss(object):

    @staticmethod
    def fn(active_func, z, y):
        return np.max(0, 1 + active_func.fn(z) - y)

    @staticmethod
    def delta(active_func, z, a, y):
        temp = active_func(z) - y
        ans = np.zeros(temp.shape)
        ans[temp >= -1] = active_func.deriv(z)*a
        return ans

#    @staticmethod
#    def fn(truth_label, predicted):
        # pos = np.sum(truth_label * predicted, axis=-1)
        # neg = np.amax((1.0-truth_label) * predicted, axis=-1)
        # return np.mean(np.max(0.0, neg - pos + 1), axis=-1)
