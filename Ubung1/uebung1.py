

from pattern import pattern1, pattern2, pattern3, pattern4
import numpy as np
import torch
import matplotlib.pyplot as plt

def convert_to_matrix(input_string):
    """
    Converts a string representation of a matrix to a numpy array
    :param input_string:
    :return: numpy array
    """
    lines = input_string.strip().split('\n')
    matrix = [[int(val) for val in line.split()] for line in lines]
    return np.array(matrix)


def forward(input_matrix, output_matrix):
    for i in range(0, 20):
        for j in range(0, 20):
            # current neutron with 1.8 wieght
            activation = 1.8 * input_matrix[i][j]
            if j - 1 >= 0:
                # left neutron with -0.1 weight
                activation += -0.1 * input_matrix[i][j - 1]
            if j + 1 <= 19:
                # right neutron with -0.1 weight
                activation += -0.1 * input_matrix[i][j + 1]
            if i - 1 >= 0:
                # top neutron with -0.1 weight
                activation += -0.1 * input_matrix[i - 1][j]
            if i + 1 <= 19:
                # bottom neutron with -0.1 weight
                activation += -0.1 * input_matrix[i + 1][j]
            if j - 1 >= 0 and i - 1 >= 0:
                # top left neutron with -0.1 weight
                activation += -0.1 * input_matrix[i - 1][j - 1]
            if j - 1 >= 0 and i + 1 <= 19:
                # bottom left neutron with -0.1 weight
                activation += -0.1 * input_matrix[i + 1][j - 1]
            if j + 1 <= 19 and i - 1 >= 0:
                # top right neutron with -0.1 weight
                activation += -0.1 * input_matrix[i - 1][j + 1]
            if j + 1 <= 19 and i + 1 <= 19:
                # bottom right neutron with -0.1 weight
                activation += -0.1 * input_matrix[i + 1][j + 1]
            output_matrix[i][j] = activation


def plot_heatmap(title, matrix):
    plt.title("Activation of the " + str(title) + " Layer")
    plt.imshow(matrix, cmap='gray')
    plt.colorbar()
    plt.show()


input_matrix = convert_to_matrix(pattern1())
output_matrix = np.zeros((20, 20), dtype=float)
forward(input_matrix, output_matrix)
plot_heatmap("Input pattern_1", input_matrix)
plot_heatmap("Output pattern_1", output_matrix)

input_matrix = convert_to_matrix(pattern2())
output_matrix = np.zeros((20, 20), dtype=float)
forward(input_matrix, output_matrix)
plot_heatmap("Input pattern_2", input_matrix)
plot_heatmap("Output pattern_2", output_matrix)

input_matrix = convert_to_matrix(pattern3())
output_matrix = np.zeros((20, 20), dtype=float)
forward(input_matrix, output_matrix)
plot_heatmap("Input pattern_3", input_matrix)
plot_heatmap("Output pattern_3", output_matrix)

input_matrix = convert_to_matrix(pattern4())
output_matrix = np.zeros((20, 20), dtype=float)
forward(input_matrix, output_matrix)
plot_heatmap("Input pattern_4", input_matrix)
plot_heatmap("Output pattern_4", output_matrix)




input_matrix = convert_to_matrix(pattern_3)


output_matrix = np.zeros((20, 20), dtype=float)

forward(input_matrix, output_matrix)

input_tensor = torch.FloatTensor(input_matrix)
output_tensor = torch.FloatTensor(output_matrix)

plot_heatmap("Input", input_matrix)
plot_heatmap("Output", output_matrix)
