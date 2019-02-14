import numpy as np


def generate_mixing_matrix(num_mixed_signals, num_source_signals):
    return np.random.rand(num_mixed_signals, num_source_signals)
