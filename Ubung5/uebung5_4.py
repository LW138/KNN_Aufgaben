import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from Ubung5.uebung5_2 import MyDataset
from Ubung5.uebung5_3 import Net

def train_model(model, train_loader, val_loader, loss_function, optimizer, num_epochs=100):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            labels = labels.view(-1,1).float()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                #outputs = (outputs > 0.5).float()
                labels = labels.view(-1, 1).float()
                loss = loss_function(outputs, labels)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

    return  train_losses, val_losses


data = MyDataset("adult/adult.data", transform=None, target_transform=None)

train_len = int(0.8 * len(data))
valid_len = int(0.1 * len(data))
test_len = len(data) - train_len - valid_len

train_dataset, valid_dataset, test_dataset = random_split(data, [train_len, valid_len, test_len])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = Net(100, 10,10,1, weight_init='xavier')
loss_func = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

train_losses, val_losses = train_model(model, train_loader, valid_loader, loss_func, optimizer)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()