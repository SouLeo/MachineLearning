import numpy as np


class CrossEntropyLoss(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(a, y):
        return a-y

# class HingeLoss(object):
#
#     @staticmethod
#     def fn(truth_label, predicted):
#         pos = np.sum(truth_label * predicted, axis=-1)
#         neg = np.max((1-truth_label) * predicted, axis=-1)
#         return np.mean(np.max(0.0, neg - pos + 1), axis=-1)
#
#     @staticmethod
#     def delta():
#         # blah