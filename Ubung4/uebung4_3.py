import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def sigmoid(x):
    return 1 / (1 + np.exp(-x / 1))


def linear(x):
    return x


# use mean_squared_error as loss function because it is a regression problem
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def predict(test_data, test_labels, weights, biases):
    correct_predictions = 0
    for i in range(len(test_data)):
        input = test_data[i]
        _, _, output = forward(input, weights, biases)
        # round the output to be able to compare it with the actual label
        predicted_label = np.round(output)
        actual_label = test_labels[i]
        #print("Predicted: ", predicted_label, "Actual: ", actual_label)
        if predicted_label == actual_label:
            correct_predictions += 1
    accuracy = correct_predictions / len(test_data)
    return accuracy


def forward(input, weights, biases):
    #calculate the activation of each layer by applying the weights and biases to the input
    layer1_act = sigmoid(np.dot(input, weights[0]) + biases[0])
    layer2_act = sigmoid(np.dot(layer1_act, weights[1]) + biases[1])
    output_act = linear(np.dot(layer2_act, weights[2]) + biases[2])
    return layer1_act, layer2_act, output_act


def backward(weights, layer1, layer2, output, expected_output):
    output_error = expected_output - output
    # delta = error * derivation of activation function => e.g: delta = (output_error) * (sigmoid(x) * (1 - sigmoid(x)))
    output_delta = output_error * 1  # derivative of linear function is 1

    # Back propagate the error of layer2 by weighting the error with the weights[2]
    layer2_error = np.dot(output_delta, weights[2].T)
    # Use the weighted error of layer2 to calculate the delta for layer2
    layer2_delta = layer2_error * layer2 * (1 - layer2)

    # Back propagate the error to layer1 by weighting the error with the weights[1]
    layer1_error = np.dot(layer2_delta, weights[1].T)
    # Use the weighted error of layer1 to calculate the delta for layer1
    layer1_delta = layer1_error * layer1 * (1 - layer1)

    return layer1_delta, layer2_delta, output_delta


def update(input, layer1, layer2, output_delta, layer1_delta, layer2_delta, weights, biases, learning_rate):
    weights[2] += learning_rate * np.dot(np.matrix(layer2).T, np.matrix(output_delta))
    biases[2] += learning_rate * output_delta

    weights[1] += learning_rate * np.dot(layer1, layer2_delta)
    biases[1] += learning_rate * layer2_delta

    weights[0] += learning_rate * np.dot(np.matrix(input).T, np.matrix(layer1_delta))
    biases[0] += learning_rate * layer1_delta


def train(training_data, training_labels, weights, biases, learning_rate, epochs):
    for j in range(epochs):
        for i in range(len(training_data)):
            input = training_data[i]
            expected_output = training_labels[i]

            # Call forward to calculate the output (activation) of each network layer (neuron)
            layer1, layer2, output = forward(input, weights, biases)

            # Use the output and apply backpropagation to calculate the delta for each layer
            layer1_delta, layer2_delta, output_delta = backward(weights, layer1, layer2, output, expected_output)

            # Update the weights and biases of the network related to the calculated deltas
            update(input, layer1, layer2, output_delta, layer1_delta, layer2_delta, weights, biases, learning_rate)

        if j % 20 == 0:
            print("Progress: ", j+20, "/", epochs, "\tLoss: ", mean_squared_error(expected_output, output))



# Read and normalize data
file_white_wine = "wine+quality/winequality-white.csv"
file_red_wine = "wine+quality/winequality-red.csv"
data_white = pd.read_csv(file_white_wine, sep=';')
data_red = pd.read_csv(file_red_wine, sep=';')

scaler = MinMaxScaler()

# leave out quality because it is the target variable
numerical_features_white = data_white.drop('quality', axis=1)
numerical_features_red = data_red.drop('quality', axis=1)

normalized_features_white = scaler.fit_transform(numerical_features_white)
normalized_features_red = scaler.fit_transform(numerical_features_red)

normalized_data_white = pd.DataFrame(normalized_features_white, columns=numerical_features_white.columns)
normalized_data_white['quality'] = data_white['quality']

normalized_data_red = pd.DataFrame(normalized_features_red, columns=numerical_features_red.columns)
normalized_data_red['quality'] = data_red['quality']

trainings_data_white = normalized_data_white.sample(frac=0.8, random_state=0)
validation_data_white = normalized_data_white.drop(trainings_data_white.index)

trainings_data_red = normalized_data_red.sample(frac=0.8, random_state=0)
validation_data_red = normalized_data_red.drop(trainings_data_red.index)

combined_data = pd.concat([trainings_data_white, trainings_data_red])
combined_data = combined_data.sample(frac=1, random_state=0) # mix data
combined_trainings_data_set = combined_data.sample(frac=0.8, random_state=0)
combined_validation_data_set = combined_data.drop(combined_trainings_data_set.index)




# Define Network
input_layer = 11
first_hidden_layer = 10
second_hidden_layer = 10
output_layer = 1

weights = [np.random.rand(input_layer, first_hidden_layer),
           np.random.rand(first_hidden_layer, second_hidden_layer),
           np.random.rand(second_hidden_layer, output_layer)]

biases = [np.zeros(first_hidden_layer), np.zeros(second_hidden_layer), np.zeros(output_layer)]
learning_rate = 0.05


# train different kinds of networks
print("Network Shape: ", input_layer, "-", first_hidden_layer, "-", second_hidden_layer, "-", output_layer)


# trains with white wine data set and predict the quality of white and red wine (exercise 4.3.1 and 4.3.2)
def network_quality_white():
    print("Length Training Data: ", len(trainings_data_white), "Length Validation Data", len(validation_data_white))

    train(trainings_data_white.drop('quality', axis=1).values, trainings_data_white['quality'].values, weights, biases,learning_rate, 5000)

    print("Accuracy on Trainings Set", predict(trainings_data_white.drop('quality', axis=1).values, trainings_data_white['quality'].values, weights, biases))
    print("Accuracy on Validation Set", predict(validation_data_white.drop('quality', axis=1).values, validation_data_white['quality'].values, weights, biases))
    print("Accuracy on Red Wine Set", predict(normalized_data_red.drop('quality', axis=1).values, normalized_data_red['quality'].values, weights, biases))


# trains with red wine data set and predict the quality of red wine (exercise 4.3.3)
def network_quality_red():
    print("Length Training Data: ", len(trainings_data_red), "Length Validation Data", len(validation_data_red))

    train(trainings_data_red.drop('quality', axis=1).values, trainings_data_red['quality'].values, weights, biases,
          learning_rate, 5000)

    print("Accuracy on Red Trainings Set",
          predict(trainings_data_red.drop('quality', axis=1).values, trainings_data_red['quality'].values, weights,
                  biases))
    print("Accuracy on Red Validation Set",
          predict(validation_data_red.drop('quality', axis=1).values, validation_data_red['quality'].values,
                  weights, biases))

# traings with a combined data set and predict the quality of white and red wine (exercise 4.3.4)
def network_quality_white_and_red():
    print("Length Training Data: ", len(combined_trainings_data_set), "Length Validation Data", len(combined_validation_data_set))

    train(combined_trainings_data_set.drop('quality', axis=1).values, combined_trainings_data_set['quality'].values, weights, biases,learning_rate, 1000)

    print("Accuracy on White Validation Set", predict(validation_data_white.drop('quality', axis=1).values, validation_data_white['quality'].values, weights, biases))
    print("Accuracy on Red Validation Set", predict(validation_data_red.drop('quality', axis=1).values, validation_data_red['quality'].values, weights, biases))
    print("Accuracy on Combined Set", predict(combined_validation_data_set.drop('quality', axis=1).values, combined_validation_data_set['quality'].values, weights, biases))



network_quality_white()
#network_quality_red()
#network_quality_white_and_red()