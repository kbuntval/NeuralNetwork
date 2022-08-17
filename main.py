import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('/Users/karanbuntval/PycharmProjects/FirstNeuralNetwork1/Data/train.csv')
data = np.array(data)
labels = data[:, 0]
data = (np.delete(data, obj=0, axis=1)).T
m, n = data.shape

print(labels)
print(m, n)

np.random.seed(42)
weights = (np.random.rand(784, 10) - 0.5).T  # Initial vector of weights
bias = np.random.rand(10, 1) - 0.5
alpha = 0.05  # learning rate


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))


inputs = data
XW = weights.dot(inputs) + bias
z = sigmoid(XW)
error = z - labels

print(z)
