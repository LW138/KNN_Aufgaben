from math import exp
from numpy import *


def sigm(x):
    return 1 / (1 + exp(-x / 1))


def forward(x, w):
    return sigm(x[0] * w[0] + x[1] * w[1] + 1 * w[2])


def predict(X, w):
    for i, item in enumerate(X):
        o = forward(X[i], w)
        print("Pattern ", i, ": ", o)
    return w


def train_and(input_patterns, labels, weights, learning_rate):
    delta_weights = [0, 0, 0]
    loss = 0.0
    for i, item in enumerate(input_patterns):
        x = input_patterns[i]
        y = labels[i]

        o = forward(x, weights)
        error = y - o
        loss += error * error

        delta_weights[0] = learning_rate * error * x[0]
        delta_weights[1] = learning_rate * error * x[1]
        delta_weights[2] = learning_rate * error * 1

        weights += delta_weights
    return weights


# create input
training_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
# expected output
labels = [0, 0, 0, 1]
# create 3 weights one for x1, x2, and bias
weights = random.rand(3)

# set learning rate
learning_rate = 0.1

# print network before training
predict(training_data, weights)
print()

# train and network and adjust weights
for i in range(1000):
    weights = train_and(training_data, labels, weights, learning_rate)
    #print(weights)
    #print("\n")

predict(training_data, weights)
