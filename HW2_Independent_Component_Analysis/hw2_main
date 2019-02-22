from scipy.io.wavfile import write
from numpy import linalg as LA
from sklearn.metrics import mean_squared_error

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from generate_mixing_matrix import generate_mixing_matrix
from cdf_guesstimate import cdf_eval

sound_mat = sio.loadmat('original_sound_data/sounds.mat')

unmixed_sounds = sound_mat['sounds']

# Generate original sound wav files
# sio.wavfile.write('original_sound_0', 44100, unmixed_sounds[0, :])
# sio.wavfile.write('original_sound_1', 44100, unmixed_sounds[1, :])
# sio.wavfile.write('original_sound_2', 44100, unmixed_sounds[2, :])
# sio.wavfile.write('original_sound_3', 44100, unmixed_sounds[3, :])
# sio.wavfile.write('original_sound_4', 44100, unmixed_sounds[4, :])

signals_to_mix = np.vstack((unmixed_sounds[0, :], unmixed_sounds[1, :], unmixed_sounds[2, :], unmixed_sounds[3, :], unmixed_sounds[4, :]))

num_of_source_signals = signals_to_mix.shape[0]
num_of_mixed_signals = 5

learning_rate = 0.01  # adjust for gradient descent

mixing_matrix = generate_mixing_matrix(num_of_mixed_signals, num_of_source_signals)
mixed_signals = np.matmul(mixing_matrix, signals_to_mix)

# scaled mixed signals as o1 and o2
o1 = np.int16(mixed_signals[0, :] / np.max(np.abs(mixed_signals[0, :])) * 32767)
o2 = np.int16(mixed_signals[1, :] / np.max(np.abs(mixed_signals[1, :])) * 32767)
o3 = np.int16(mixed_signals[2, :] / np.max(np.abs(mixed_signals[2, :])) * 32767)
o4 = np.int16(mixed_signals[3, :] / np.max(np.abs(mixed_signals[3, :])) * 32767)
o5 = np.int16(mixed_signals[4, :] / np.max(np.abs(mixed_signals[4, :])) * 32767)
# Generates mixed sound wav file
# sio.wavfile.write('mixed_sound_data/mixed_sound_0_1', 44100, mixed_signals_scaled)

# Create W Matrix (n by m) to recover the original n source signals
W = 2*np.random.rand(num_of_source_signals, num_of_mixed_signals)
# TODO: Create Gradient Descent Loop Here
for i in range(1000000):
    # Step 3
    curr_source_estimate = np.matmul(W, mixed_signals)
    # Step 4
    Z = cdf_eval(curr_source_estimate)
    # Step 5
    del_W = 1-2*Z
    del_W = (1/44000)*np.matmul(del_W, curr_source_estimate.transpose())  # 1/44000*
    del_W = np.identity(num_of_source_signals) + del_W
    del_W = del_W * learning_rate
    del_W = np.matmul(del_W, W)
    # Step 6
    W = W + del_W
    if LA.norm(del_W) < 0.01:
        print('converged')
        break

# Save reconstructed sounds to unmixed_sound_data directory
t1 = np.int16(curr_source_estimate[0, :] / np.max(np.abs(curr_source_estimate[0, :])) * 32767)

t2 = np.int16(curr_source_estimate[1, :] / np.max(np.abs(curr_source_estimate[1, :])) * 32767)

t3 = np.int16(curr_source_estimate[2, :] / np.max(np.abs(curr_source_estimate[2, :])) * 32767)

t4 = np.int16(curr_source_estimate[3, :] / np.max(np.abs(curr_source_estimate[3, :])) * 32767)

t5 = np.int16(curr_source_estimate[4, :] / np.max(np.abs(curr_source_estimate[4, :])) * 32767)

# map original signals, mixed signals, and reconstructed signals between 0 and 1 for plotting
# original signals:
original_signals_scaled_1 = signals_to_mix[0, :] / np.linalg.norm(signals_to_mix[0, :])
original_signals_scaled_2 = signals_to_mix[1, :] / np.linalg.norm(signals_to_mix[1, :])
original_signals_scaled_3 = signals_to_mix[2, :] / np.linalg.norm(signals_to_mix[2, :])
original_signals_scaled_4 = signals_to_mix[3, :] / np.linalg.norm(signals_to_mix[3, :])
original_signals_scaled_5 = signals_to_mix[4, :] / np.linalg.norm(signals_to_mix[4, :])

# mixed signals: o
o1 = o1 / np.linalg.norm(o1)
o2 = o2 / np.linalg.norm(o2)
o3 = o3 / np.linalg.norm(o3)
o4 = o4 / np.linalg.norm(o4)
o5 = o5 / np.linalg.norm(o5)

# recovered signals: t
t1_scaled = t1 / np.linalg.norm(t1)
t2_scaled = t2 / np.linalg.norm(t2)
t3_scaled = t3 / np.linalg.norm(t3)
t4_scaled = t4 / np.linalg.norm(t4)
t5_scaled = t5 / np.linalg.norm(t5)

# Cross correlation/ signal assignment logic
originals = np.vstack((original_signals_scaled_1, original_signals_scaled_2, original_signals_scaled_3, original_signals_scaled_4, original_signals_scaled_5))
reconstructed = np.vstack((t1_scaled, t2_scaled, t3_scaled, t4_scaled, t5_scaled))

Dict = {}

for i in range(num_of_source_signals):
    temp_max = 0
    temp_index = 0
    for j in range(num_of_mixed_signals):
        curr_max = np.absolute(np.correlate(originals[i, :], reconstructed[j, :]))
        # print(curr_max)
        if curr_max > temp_max and j not in Dict.values():
            temp_max = curr_max
            temp_index = j
    Dict[i] = temp_index
    # print(mean_squared_error(originals[i, :], reconstructed[temp_index, :]))
print(Dict)

A = np.array([[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)], [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)], [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)], [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)], [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]], dtype=object)
for i in range(num_of_source_signals):
    for j in range(num_of_mixed_signals):
        corr = np.correlate(originals[i, :], reconstructed[j, :])
        mse = mean_squared_error(originals[i, :], reconstructed[j, :])
        A[i][j] = (corr, mse)
print(A)

# # insert accuracy determination (MSE) within here
# for i in range(num_of_source_signals):
    # mse = mean_squared_error(originals[i, :], reconstructed[Dict[i], :])
    # print(mse)

# begin plotting signals
fig, axs = plt.subplots(3, 1)
samp_number = np.arange(44000)
offset = 0.1
# original signals
axs[0].plot(samp_number, original_signals_scaled_1, 'r', samp_number, original_signals_scaled_2 + offset, 'b', samp_number, original_signals_scaled_3 + 2*offset, 'g', samp_number, original_signals_scaled_4 + 3*offset, 'y', samp_number, original_signals_scaled_5 + 4*offset, 'm')
axs[0].set_xlabel('original samples')

# mixed signals
axs[1].plot(samp_number, o1, samp_number, o2 + offset, samp_number, o3 + 2*offset, samp_number, o4 + 3*offset, samp_number, o5 + 4*offset)
axs[1].set_xlabel('mixed samples')

# unmixed signals plot
axs[2].plot(samp_number, reconstructed[Dict[0], :], 'r', samp_number, reconstructed[Dict[1], :] + offset, 'b', samp_number, reconstructed[Dict[2], :] + 2*offset, 'g', samp_number, reconstructed[Dict[3], :] + 3*offset, 'y', samp_number, reconstructed[Dict[4], :] + 4*offset, 'm')
axs[2].set_xlabel('reconstructed samples')

plt.show()
print('done')
