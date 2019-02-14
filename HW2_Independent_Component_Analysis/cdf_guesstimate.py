import numpy as np


def cdf_eval(x):
    return 1/(1+np.exp(-x))


cdf_eval = np.vectorize(cdf_eval, otypes=[np.float64])
