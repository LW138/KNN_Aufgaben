import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.datasets as datasets
from torchvision import transforms
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy


class FashionMNISTNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, weight_init='xavier'):
        super(FashionMNISTNet, self).__init__()
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

    def forward(self, input):
        # F is an alias for torch.nn.functional, which contains functions for many common operations on tensors
        # Flatten the input tensor
        input_flatten = input.view(input.size(0), -1)
        out_layer1 = F.relu(self.layer1(input_flatten))
        out_layer2 = F.relu(self.layer2(out_layer1))
        out_layer3 = F.softmax(self.layer3(out_layer2), dim=1)
        return out_layer3

    def train_model(self, train_loader, valid_loader, optimizer, loss_function, logger, num_epochs, device='cpu'):
        # Set the model to training mode
        self.to(device)
        self.train()
        train_accuracy = Accuracy(task='multiclass', num_classes=10, average='micro').to(device)
        valid_accuracy = Accuracy(task='multiclass', num_classes=10, average='micro').to(device)

        # Loop over the epochs
        for epoch in range(num_epochs):
            train_loss = 0.0
            train_accuracy.reset()

            for i, data in enumerate(train_loader, 0):
                # Get the inputs and labels
                inputs, labels = data
                # Move inputs and labels to the specified device
                inputs, labels = inputs.to(device), labels.to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = self(inputs)
                # Calculate the loss
                loss = loss_function(outputs, labels)
                # Backward pass
                loss.backward()
                # Update the weights
                optimizer.step()
                # Add the loss to the running_loss
                train_loss += loss.item()
                # Update accuracy
                train_accuracy.update(outputs, labels)

            train_loss = train_loss / len(train_loader)
            train_acc = train_accuracy.compute()
            logger.add_scalar('Loss/train', train_loss, epoch + 1)
            logger.add_scalar('Accuracy/train', train_acc, epoch + 1)

            # Set the model to evaluation mode
            self.eval()
            val_loss = 0.0
            valid_accuracy.reset()

            with torch.no_grad():
                for val_input, val_label in valid_loader:
                    val_input, val_label = val_input.to(device), val_label.to(device)
                    outputs = self(val_input)
                    loss = loss_function(outputs, val_label)
                    val_loss += loss.item()
                    # Update accuracy
                    valid_accuracy.update(outputs, val_label)

            val_loss = val_loss / len(valid_loader)
            val_acc = valid_accuracy.compute()
            logger.add_scalar('Loss/validate', val_loss, epoch + 1)
            logger.add_scalar('Accuracy/validate', val_acc, epoch + 1)
            logger.flush()

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

    def check_test_accuracy(self, test_loader, device='cpu'):
        self.eval()
        test_accuracy = Accuracy(task='multiclass', num_classes=10, average='micro').to(device)

        with torch.no_grad():
            for test_input, test_label in test_loader:
                test_input, test_label = test_input.to(device), test_label.to(device)
                outputs = self(test_input)
                _, predicted = torch.max(outputs.data, 1)
                test_accuracy.update(predicted, test_label)

        acc = test_accuracy.compute()
        print(f"Accuracy on the test set: {acc * 100}%")


if __name__ == "__main__":
    # Define a transform to convert the images to tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)

    train_len = int(0.8 * len(dataset))
    valid_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - valid_len
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_len, valid_len, test_len])

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    loss_func = nn.CrossEntropyLoss()

    model = FashionMNISTNet(784, 16, 16, 10, weight_init='xavier')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    logger = SummaryWriter()

    model.train_model(train_loader, valid_loader, optimizer, loss_func, logger, 25, device=device)

    accuracy = Accuracy(task='multiclass', num_classes=10, average='micro')
    model.check_test_accuracy(test_loader, device=device)

    logger.close()

    """
    normal (model = FashionMNISTNet(784, 16, 16, 10,  weight_init='xavier')): 
        Epoch 50/50, Train Loss: 1.7269659439722698, Validation Loss: 1.7293099164962769
        Accuracy on the test set: 72.86666666666666%
    overfitting (model = FashionMNISTNet(784, 750, 750, 10,  weight_init='xavier')):
        Epoch 50/50, Train Loss: 2.361170689264933, Validation Loss: 2.3603177070617676
        Accuracy on the test set: 9.95%
    underfitting (model = FashionMNISTNet(784, 1, 1, 10,  weight_init='xavier')):
        Epoch 50/50, Train Loss: 2.1603426933288574, Validation Loss: 2.1632094383239746
        Accuracy on the test set: 27.633333333333333%
    """
