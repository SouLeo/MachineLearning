import numpy as np
from inputData import mat_input
import activationfuncs


class FFNetwork(object):

    def __init__(self, layers_and_dim):
        self.layers_and_dim = layers_and_dim
        self.num_layers = len(self.layers_and_dim)
        self.num_neurons = np.sum(self.layers_and_dim)
        # for MNIST, input must always be 784 and output must always be 10
        if self.layers_and_dim[0] != 784 or self.layers_and_dim[-1] != 10:
            print('WARNING: INPUT AND OUTPUT LAYER DIM. SET INAPPROPRIATELY.')
            print('INPUT ADJUSTED TO 784. OUTPUT ADJUSTED TO 10.')
            self.layers_and_dim[0] = 784
            self.layers_and_dim[-1] = 10
        # randomly generate biases and weights for every neuron
        # self.biases size is layer width by num_layers (starting from first hidden)
        self.biases = [np.random.randn(x, 1) for x in self.layers_and_dim[1:]]
        # self.weights is
        shift = np.roll(self.layers_and_dim, 1)
        weight_combos = np.column_stack((self.layers_and_dim, shift))
        weight_combos = np.delete(weight_combos, 0, 0)
        self.weights = [np.random.randn(x, y) for x, y in weight_combos]
        # print(self.weights[0].shape)
        #self.cost = cost

    def create_mini_batch(self, train_img, train_labels, mini_batch_size):
        mini_batch_img = [train_img[:, x:x+mini_batch_size]
                          for x in range(0, train_img.shape[1], mini_batch_size)]
        mini_batch_label = [train_labels[x:x+mini_batch_size]
                            for x in range(0, len(train_labels), mini_batch_size)]
        # print(mini_batch_img[0].shape)
        # print(len(mini_batch_label[0]))
        return mini_batch_img, mini_batch_label

    def stoch_grad_desc(self, epochs, learn_rate, mini_batch_img, mini_batch_labels):
        # print(mini_batch_img[2].shape) # length is the size of mini_batch
        # print(mini_batch_img[2][:, 1]) # shape: (784, 1)
        # print(len(mini_batch_labels[2])) # length is size of mini_batch
        # print(mini_batch_labels[2][1]) # shape: (10, 1)
        for i in range(epochs):
            for m in range(len(mini_batch_labels)):
                self.update_params(mini_batch_img[m], mini_batch_labels[m], learn_rate)
                #print(m)  # length of for loop is 60,000 / mini_batch_size

    def update_params(self, img, label, learn_rate):
        # img shape: (784, mini_batch_size) <- (784, 100)
        # label len: (100) <- list NOT vec
        # print(img[:, 1].shape)
        # print(len(label[1]))
        #
        dL_db = [np.zeros(b.shape) for b in self.biases]
        dL_dw = [np.zeros(w.shape) for w in self.weights]
        # print(dL_db[2].shape)
        # print(dL_dw[2].shape)
        for i in range(len(label)):
            self.backpropagation(img[:, i], label[i])
            # TODO: finish writing for loop

    def backpropagation(self, img, label):
        # img shape (784, 1)
        # label shape (10, 1)
        dL_db = [np.zeros(b.shape) for b in self.biases]
        dL_dw = [np.zeros(w.shape) for w in self.weights]
        #
        activation = img
        activation_list = [img]
        z_s = []  # preactivation func. vectors (list z's)

        # print(len(self.biases))
        # print(len(self.weights))
        # print(self.weights[0].shape)

        for i in range(len(self.biases)):
            # print(w.shape)
            z = np.dot(self.weights[i], activation) + self.biases[i]
            z_s.append(z)
            activation = activationfuncs.sigmoid(z)  #overflows
            activation_list.append(activation)

        # backprop algo starts:


def main():
    # NN Arch Params
    epochs = 4
    learn_rate = 0.5
    mini_batch_size = 100
    nn_architecture = np.array([784, 16, 16, 10])

    test = FFNetwork(nn_architecture)
    # TODO: Put following code into a ffnetwork member func.
    train_images, train_labels, test_images, test_labels = mat_input()
    mini_img, mini_lab = test.create_mini_batch(train_images, train_labels, mini_batch_size)
    test.stoch_grad_desc(epochs, learn_rate, mini_img, mini_lab)


if __name__ == '__main__':
    main()
