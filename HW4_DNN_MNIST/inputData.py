import numpy as np
import scipy.io as sio


def mat_input():
    mnist_data = sio.loadmat('digits.mat')

    train_labels = mnist_data['trainLabels']
    test_labels = mnist_data['testLabels']
    test_images = mnist_data['testImages']
    train_images = mnist_data['trainImages']

    train_images = np.reshape(train_images, (1, 784, 1, 60000))
    train_images = np.squeeze(train_images)

    train_labels = np.squeeze(train_labels)
    train_labels = [one_hot(y) for y in train_labels]

    test_images = np.reshape(test_images, (1, 784, 1, 10000))
    test_images = np.squeeze(test_images)

    return(train_images, train_labels, test_images, test_labels)


def one_hot(i):
    vec = np.zeros((10, 1))
    vec[i] = 1.0
    return vec


# def main():
#     mat_input()
#
#
# if __name__ == '__main__':
#     main()
