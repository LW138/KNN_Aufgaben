import numpy as np
import tensorflow as tf

def sigmoid(x):
    return 1 / (1 + np.exp(-x / 1))

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.sum(y_true * np.log(y_pred))

def predict(test_images, test_labels, weights, biases):
    correct_predictions = 0
    for i in range(len(test_images)):
        input = (test_images[i] / 255).flatten()
        _, _, output = forward(input, weights, biases)
        predicted_label = np.argmax(output)
        actual_label = test_labels[i]
        print("Predicted: ", predicted_label, "Actual: ", actual_label)
        if predicted_label == actual_label:
            correct_predictions += 1
    accuracy = correct_predictions / len(test_images)
    return accuracy


def forward(input, weights, biases):
    # 784x1 * 784x256 = 256x1
    layer1_act = sigmoid(np.dot(input, weights[0]) + biases[0])
    # 256x1 * 256x256 = 256x1
    layer2_act = sigmoid(np.dot(layer1_act, weights[1]) + biases[1])
    # 256x1 * 256x10 = 10x1
    output_act = sigmoid(np.dot(layer2_act.T, weights[2]) + biases[2])

    return layer1_act, layer2_act, output_act


def backward(weights, layer1, layer2, output, expected_output):
    # Calculate the error of the output layer
    output_error = expected_output - output
    # delta = error - derivation of activation function => delta = output_error * sigmoid(x) * (1 - sigmoid(x))
    output_delta = output_error * output * (1 - output)

    # Back propagate the error to layer2 by weighting the error with the weights[2]
    layer2_error = np.dot(output_delta, weights[2].T)
    # Use the weighted error of layer2 to calculate the delta for layer2
    layer2_delta = layer2_error * layer2 * (1 - layer2)

    # Back propagate the error to layer1 by weighting the error with the weights[1]
    layer1_error = np.dot(layer2_delta, weights[1].T)
    # Use the weighted error of layer1 to calculate the delta for layer1
    layer1_delta = layer1_error * layer1 * (1 - layer1)

    return layer1_delta, layer2_delta, output_delta


def update(input, layer1, layer2, output_delta, layer1_delta, layer2_delta, weights, biases, learning_rate):
    # weights[2] = learning_rate * (256x1 * 10x1) = 256x10
    weights[2] += learning_rate * np.outer(layer2, output_delta)
    biases[2] += learning_rate * output_delta

    # weights[1] = learning_rate * (256x1 * 256x1) = 256x256
    weights[1] += learning_rate * np.outer(layer1, layer2_delta)
    biases[1] += learning_rate * layer2_delta

    # weights[0] = learning_rate * (784x1 * 256x1) = 784x256
    weights[0] += learning_rate * np.outer(input,layer1_delta)
    biases[0] += learning_rate * layer1_delta


def train(train_images, train_labels, weights, biases, learning_rate, epochs):
    for j in range(epochs):
        for i in range(len(train_images)):
            # Normalize image values (0 to 255) in a range from 0 to 1 and convert the matrix from 28x28 to 784x1
            input = (train_images[i] / 255).flatten()

            # Create a one hot encoded vector for the output layer
            expected_output = np.zeros(output_layer)
            expected_output[train_labels[i]] = 1

            # Call forward to calculate the output (activation) of each network layer (neuron)
            layer1, layer2, output = forward(input, weights, biases)

            # Use the output and apply backpropagation to calculate the delta for each layer
            layer1_delta, layyer2_delta, output_delta = backward(weights, layer1, layer2, output, expected_output)

            # Update the weights and biases of the network related to the calculated deltas
            update(input, layer1, layer2, output_delta, layer1_delta, layyer2_delta, weights, biases, learning_rate)


        if j % 10 == 0:
            print("Progress: ", j+10, "/", epochs, "\tLoss: ", cross_entropy(expected_output, output))


learning_rate = 0.1

# Create a network of shape 784 - 256 - 256 - 10
input_layer = 784
first_hidden_layer = 16 #256
second_hidden_layer = 16 #256
output_layer = 10

weights = [np.random.rand(input_layer, first_hidden_layer) * 0.01, # * 0.01 to keep the weights small
           np.random.rand(first_hidden_layer, second_hidden_layer) * 0.01,
           np.random.rand(second_hidden_layer, output_layer) * 0.01]

biases = [np.zeros(first_hidden_layer), np.zeros(second_hidden_layer), np.zeros(output_layer)]

# Load the mnist dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Train the network with the first 10000 images for a faster training
train(train_images[:10000], train_labels[:10000], weights, biases, learning_rate, 100)


print("Accuracy: ", predict(test_images[:100], test_labels[:100], weights, biases))

"""
1. Welche Accuracy kann mit 784 - 256 - 256 - 10 erreicht werden?
    ~99% Accuracy geschätzt (loss war 0,0003, aber nicht fertig trainiert). Bei Loss = 1,6 bereits 70% Accuracy
    
2. Welche Auswirkung hat eine Vergrößerung des Netzes?
    Die Accuracy steigt, da das Netz mehr Parameter hat, um die Daten zu lernen. Allerdings gibt es ein Overfitting, wenn das Netz zu groß ist.
    Allerdings steigt auch die Rechenzeit und der Speicherbedarf.
    
3. Was ist das kleinste Netz, das noch funktioniert?
    Getestet mit: Learning Rate: 0.1, Epochs: 100, Trainingsdaten: 10000, Testdaten: 100
    784 - 256 - 256 - 10 => Accuracy: 97% Loss: 0,00047
    784 - 128 - 128 - 10 => Accuracy: 97%
    784 - 64 - 64 - 10 => Accuracy: 96%
    784 - 32 - 32 - 10 => Accuracy: 96% Loss: 0,0059
    784 - 16 - 16 - 10 => Accuracy: 89% Loss 0,0039
    784 - 8 - 8 - 10 => Accuracy: 90%
    784 - 4 - 4 - 10 => Accuracy: 79%
    784 - 2 - 2 - 10 => Accuracy: 23%
"""
