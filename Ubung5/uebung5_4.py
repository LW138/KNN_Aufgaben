import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from Ubung5.uebung5_2 import MyDataset
from Ubung5.uebung5_3 import Net


def train_model(model, train_loader, val_loader, loss_function, optimizer, num_epochs=50, device='cpu'):
    # Move model to the specified device
    model.to(device)

    # Lists to store the losses to plot them afterwards
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        # Initialize the running loss for the current epoch to 0
        running_loss = 0.0

        # Iterate over the training data
        for train_input, train_label in train_loader:
            # Move inputs and labels to the specified device
            train_input, train_label = train_input.to(device), train_label.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(train_input)

            # Calculate the loss
            train_label = train_label.view(-1, 1).float()
            loss = loss_function(output, train_label)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Set the model to evaluation mode
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_input, val_label in val_loader:
                # Move inputs and labels to the specified device
                val_input, val_label = val_input.to(device), val_label.to(device)

                outputs = model(val_input)
                val_label = val_label.view(-1, 1).float()
                loss = loss_function(outputs, val_label)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

    return train_losses, val_losses

def check_test_accuracy(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for test_input, test_label in test_loader:
            test_input, test_label = test_input.to(device), test_label.to(device)
            outputs = model(test_input)
            predicted = (outputs > 0.5).float()
            total += test_label.size(0)
            correct += (predicted.view(-1) == test_label.float()).sum().item()

    print(f"Accuracy on the test set: {100 * correct / total}%")


# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data = MyDataset("adult/adult.data", transform=None, target_transform=None)

train_len = int(0.8 * len(data))
valid_len = int(0.1 * len(data))
test_len = len(data) - train_len - valid_len

train_dataset, valid_dataset, test_dataset = random_split(data, [train_len, valid_len, test_len])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = Net(100, 10, 10, 1, weight_init='kaiming')
loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.1)

train_losses, val_losses = train_model(model, train_loader, valid_loader, loss_func, optimizer, device=device)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

check_test_accuracy(model, test_loader, device)
