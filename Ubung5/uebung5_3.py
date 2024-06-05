import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, weight_init='xavier'):
        super(Net, self).__init__()

        # Create three linear layers
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, num_classes)

        #  based on the weight_init parameter, the weights of the linear layers are initialized
        if weight_init == 'xavier':
            init.xavier_uniform_(self.layer1.weight)
            init.xavier_uniform_(self.layer2.weight)
            init.xavier_uniform_(self.layer3.weight)
        elif weight_init == 'kaiming':
            init.kaiming_uniform_(self.layer1.weight, nonlinearity='relu')
            init.kaiming_uniform_(self.layer2.weight, nonlinearity='relu')
            init.kaiming_uniform_(self.layer3.weight, nonlinearity='relu')

        init.zeros_(self.layer1.bias)
        init.zeros_(self.layer2.bias)
        init.zeros_(self.layer3.bias)

    def forward(self, x):
        # F in the code is an alias for torch.nn.functional,
        # which is a module in PyTorch that contains functions for many common operations on tensor
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = torch.sigmoid(self.layer3(out))
        return out


if __name__ == "__main__":

    # create an instance of the Net class
    model = Net(14, 10, 10, 1, weight_init='xavier')

    # create a random tensor with 16 samples and 14 features (14 because of the input size of the model)
    x = torch.rand(16, 14)

    # Forward-Funktion aufrufen
    output = model.forward(x)

    print(output)