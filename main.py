import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data =

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)  # If test data is provided, progress will be printed out after each epoch
        n = len(training_data)
        for j in range(epochs):
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1}/{2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            delta = (activations[-1] - y) * \
                    sigmoid_prime(zs[-1])
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            # Note that the variable l in the loop below is used a little
            # differently to the notation in Chapter 2 of the book.  Here,
            # l = 1 means the last layer of neurons, l = 2 is the
            # second-last layer, and so on.  It's a renumbering of the
            # scheme in the book, used here to take advantage of the fact
            # that Python can use negative indices in lists.
            for l in range(2, self.num_layers):
                z = zs[-l]
                sp = sigmoid_prime(z)
                delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

#
# data = pd.read_csv('/Users/karanbuntval/PycharmProjects/FirstNeuralNetwork1/Data/train.csv')
# data = np.array(data)
# Y_label = data[:, 0]  # Labels within dataset (42000 * 1)
# A1 = (np.delete(data, obj=0, axis=1)).T  # Pixel values w/o corresponding labels (784 * 42000)
# m, n = data.shape
#
# def params():
#     np.random.seed(1)
#     w1 = (np.random.rand(784, 10) - 0.5).T  # Initial vector of weights (10 * 784)
#     b1 = np.random.rand(10, 1) - 0.5
#     w2 = (np.random.rand(10, 10) - 0.5)
#     b2 = np.random.ran(10, 1) - 0.5
#     alpha = 0.05  # learning rate
#     return w1, b1, w2, b2, alpha
#
#
# def relu(x):
#     return np.maximum(0, x)
#
#
# def softmax(x):
#     return np.exp(x)/sum(np.exp(x))
#
#
# def forward_prop(inp, w1, b1, w2, b2):
#     Z1 = w1.dot(inp) + b1
#     A1 = relu(Z1)
#     Z2 = w2.dot(A1) + b2
#     A2 = softmax(Z2)
#     return Z1, A1, Z2, A2
#
#
# def one_hot(Y):
#     one_hot_Y = np.zeros((Y.size, Y.max() + 1))
#     one_hot_Y[np.arange(Y.size), Y] = 1
#     one_hot_Y = one_hot_Y.T
#     return one_hot_Y
#
# def backward_prop()

