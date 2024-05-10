import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from math import exp


# load the shapes from the shapes directory
def load_inputs():
    prefix = "shapes/"
    minus = np.load(prefix + "minus.npy")
    plus = np.load(prefix + "plus.npy")
    ball = np.load(prefix + "ball.npy")
    divide = np.load(prefix + "divide.npy")
    times = np.load(prefix + "times.npy")
    inputs = [minus, plus, ball, divide, times]
    return inputs


# plot the inputs as a heatmap
def plot_inputs(inputs):
    for shape in inputs:
        plt.imshow(shape, cmap='gray')
        plt.show()


def sigm(x):
    return 1 / (1 + exp(-x / 1))


def forward(input_vector, weights, biases):
    output = [0.0, 0.0, 0.0, 0.0, 0.0]
    act_sum = 0.0
    for i in range(len(weights)):
        for j in range(len(weights[0])):
            act_sum += input_vector[j] * weights[i][j]
        act_sum += biases[i]
        output[i] = sigm(act_sum)
        act_sum = 0.0
    return output


def predict(input, weights, biases):
    outputs = []
    for i, item in enumerate(input):
        o = forward(item, weights, biases)
        winner = np.argmax(o)
        print("\nPattern ", i, ": Winner is ", winner, " with an activation of: ", o[winner])
        print("Activation of all output neurons: ", o)
        outputs.append(o)


def train(input_patterns, labels, weights, biases, learning_rate):
    delta_weights = np.zeros((5, 25))
    delta_biases = np.zeros(5)
    error = np.zeros(5)

    for i in range(len(input_patterns)):
        x = input_patterns[i]
        y = labels[i]
        o = forward(x, weights, biases)

        for i in range(len(y)):
            error[i] += y[i] - o[i]

        for j in range(len(weights)):  # length of output weights
            for i in range(len(weights[j])):  # length of weights[0]
                if x[i]:
                    delta_weights[j][i] = (learning_rate * error[j])
            delta_biases[j] += learning_rate * error[j]

    weights += delta_weights
    biases += delta_biases

    return weights, biases


# load and visualize the shapes
inputs = load_inputs()
# plot_inputs(inputs)

weights = np.random.randn(5, 25)  # normalverteilte Zufallszahlen (5x25)
biases = np.zeros(5)

# flatten the input matrix to a vector
input_vectors = []
for i in range(len(inputs)):
    input_vectors.append(inputs[i].flatten())

# defines the labels for the given shapes
labels = [[1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 1.0]]

# define learning_rate
learning_rate = 0.1

# train the network 50 times with a training batch size of 10 pattern for each iteration
for i in range(50):
    weights, biases = train(input_vectors * 2, labels * 2, weights, biases, learning_rate)

# predict the input shapes
predict(input_vectors, weights, biases)