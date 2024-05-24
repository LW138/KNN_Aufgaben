import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, weight_init='xavier'):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, num_classes)

        # Gewichtsinitialisierung
        if weight_init == 'xavier':
            init.xavier_uniform_(self.layer1.weight)
            init.xavier_uniform_(self.layer2.weight)
            init.xavier_uniform_(self.layer3.weight)
        elif weight_init == 'kaiming':
            init.kaiming_uniform_(self.layer1.weight, nonlinearity='relu')
            init.kaiming_uniform_(self.layer2.weight, nonlinearity='relu')
            init.kaiming_uniform_(self.layer3.weight, nonlinearity='relu')

    def forward(self, x):
        # F in the code is an alias for torch.nn.functional, which is a module in PyTorch that contains functions for many common operations on tensor
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.sigmoid(self.layer3(out))
        return out


if __name__ == "__main__":

    # Modellinstanz erstellen
    model = Net(14, 10, 10, 2, weight_init='xavier')

    # Zufälligen Tensor erstellen
    x = torch.rand(16, 14)

    # Forward-Funktion aufrufen
    output = model.forward(x)

    print(output)