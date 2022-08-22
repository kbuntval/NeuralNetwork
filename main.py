import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def sigmoid(z):
    return 1/(1 + np.exp(-z))


net = Network([2, 3, 1])

print(net.weights[])


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

