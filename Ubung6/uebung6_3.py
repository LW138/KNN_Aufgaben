import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.datasets as datasets
from torchvision import transforms
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split


class FashionMNISTNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, weight_init='xavier'):
        super(FashionMNISTNet, self).__init__()
        # Create three linear layers
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, num_classes)
        ##### Flat Spot Optimierer Dropout ######
        self.dropout = nn.Dropout(0.5)
        ##### Flat Spot Optimierer Batch Normalization ######
      #  self.bn1 = nn.BatchNorm1d(hidden_size1)
      #  self.bn2 = nn.BatchNorm1d(hidden_size2)

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
        # F is an alias for torch.nn.functional, which contains functions for many common operations on tensors
        # Flatten the input tensor
        x = x.view(x.size(0), -1)
        out = F.relu(self.layer1(x))
        out = self.dropout(out)
        out = F.relu(self.layer2(out))
        out = self.dropout(out)
        out = F.softmax(self.layer3(out), dim=1)
        return out

    def train_model(self, train_loader, valid_loader, optimizer, loss_function, scheduler, logger, num_epochs=20, early_stopping=False, device='cpu'):
        # redirct input to GPU if available and set the model to training mode
        self.to(device)

        # this mode handling is critical to ensure that the dropout is only applied during training
        self.train()

        # Variables need for early stopping improvement
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        # Loop over the epochs
        for epoch in range(num_epochs):
            running_loss = 0.0
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
                running_loss += loss.item()
            train_loss = running_loss / len(train_loader)
            logger.add_scalar('Loss/train', train_loss, epoch+1)

            # Set the model to evaluation mode to disable droput
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_input, val_label in valid_loader:
                    val_input, val_label = val_input.to(device), val_label.to(device)
                    outputs = model(val_input)
                    loss = loss_function(outputs, val_label)
                    val_loss += loss.item()
            val_loss = val_loss / len(valid_loader)
            logger.add_scalar('Loss/validate', val_loss, epoch+1)
            logger.flush()
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
            scheduler.step()

            ##### Flat Spot Optimierer Early Stopping ######
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("Early stopping")
                    break

    def get_initial_loss(self, train_loader, valid_loader, loss_function, device='cpu'):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self(inputs)
            loss = loss_function(outputs, labels)
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        logger.add_scalar('Loss/train', train_loss, 0)

        val_loss = 0.0
        for val_input, val_label in valid_loader:
            val_input, val_label = val_input.to(device), val_label.to(device)
            outputs = model(val_input)
            loss = loss_function(outputs, val_label)
            val_loss += loss.item()
        val_loss = val_loss / len(valid_loader)
        logger.add_scalar('Loss/validate', val_loss, 0)
        logger.flush()
        print(f"Initial Loss, Train Loss: {train_loss}, Validation Loss: {val_loss}")

    def check_test_accuracy(self, test_loader, device='cpu'):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for test_input, test_label in test_loader:
                test_input, test_label = test_input.to(device), test_label.to(device)
                outputs = self(test_input)
                _, predicted = torch.max(outputs.data, 1)
                total += test_label.size(0)
                correct += (predicted == test_label).sum().item()

        print(f"Accuracy on the test set: {100 * correct / total}%")

if __name__ == "__main__":
    logger = SummaryWriter()
    # Define a transform to convert the images to tensors and add data augmentation

    ####### Flat Spot Optimierer Augmentation ######
    transform = transforms.Compose([
      #  transforms.RandomHorizontalFlip(),
       # transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
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

    model = FashionMNISTNet(784, 16, 16, 10, weight_init='xavier')
    loss_func = nn.CrossEntropyLoss()

    # Define the several optimizers
    ##### Flat Spot Optimierer Weight Decay ######
    SGD = optim.SGD(model.parameters(), lr=0.01, momentum=0.8, weight_decay=0.0005)
    Adam = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    RMSprop = optim.RMSprop(model.parameters(), lr=0.01, weight_decay=0.0005)
    optimizer = SGD

    ##### Flat Spot Optimierer warmup ######
    warmup_steps = 10
    lr_lambda = lambda epoch: epoch / warmup_steps if epoch < warmup_steps else 1
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Train the model. 1. get initial loss 2. train the model 3. check the test accuracy
    model.get_initial_loss(train_loader, valid_loader, loss_func, device=device)
    model.train_model(train_loader, valid_loader, optimizer, loss_func, scheduler, logger,  num_epochs=25, early_stopping=True, device=device)
    model.check_test_accuracy(test_loader, device=device)

    logger.close()
