from math import exp
from numpy import *

def sigm(x):
    return 1 / (1 + exp(-x / 1))

def forward(x, w):
  return sigm(x[0] * w[0] + 1 * w[1])

def predict(X,w):
  for i, item in enumerate(X):
    o = forward(X[i],w)
    print("Pattern ", i, ": ", o)
  return w


def train_not(input_patterns, labels, weights, learning_rate):
  delta_weights = [0, 0]
  loss = 0.0

  for i, item in enumerate(input_patterns):
    x = input_patterns[i]
    y = labels[i]

    o = forward(x, weights)
    error = y - o
    loss += error * error

    delta_weights[0] = learning_rate * error * x[0]
    delta_weights[1] = learning_rate * error * 1

    weights += delta_weights

  return weights


#create input data set
training_data = [[0],[1]]

#define expected output labels
labels = [1,0]

#create 2 weights. 1 for bias, 1 for neuron
weights = random.rand(2)

#set learning rate (0.1 is standard range)
learning_rate = 0.1


# print network before training
predict(training_data, weights)
print()

#train not network and adjust weights
for i in range(1000):
    weights = train_not(training_data, labels, weights, learning_rate)
    #print(weights)
    #print("\n")

predict(training_data, weights)