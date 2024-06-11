import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import transforms


class LeNet(nn.Module):
    """The LeNet-5 model."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.net = nn.Sequential(
            #use this for FashionMNIST: nn.Conv2d(1,6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.Conv2d(3,6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6,16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            #Use this for FashionMNIST (other input dimension): nn.Linear(400, 64), nn.ReLU(),
            nn.Linear(576, 256), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64, nn.ReLU()),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes, nn.Softmax(dim=1))
        )
        self.init_cnn()

    def init_cnn(self):
        """Initialize weights for CNNs."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.net(x)

    def train_model(self, train_loader, valid_loader, optimizer, loss_func, scheduler, logger, num_epochs=25, early_stopping=True, device='cpu'):
        self.to(device)
        self.train()
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss / len(train_loader)
            logger.add_scalar('Loss/train', train_loss, epoch+1)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_input, val_label in valid_loader:
                    val_input, val_label = val_input.to(device), val_label.to(device)
                    outputs = self(val_input)
                    loss = loss_func(outputs, val_label)
                    val_loss += loss.item()
            val_loss = val_loss / len(valid_loader)
            logger.add_scalar('Loss/validate', val_loss, epoch+1)
            logger.flush()
            scheduler.step()
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("Early stopping")
                    break


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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))
    logger = SummaryWriter()
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data = datasets.CIFAR10('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    train_len = int(0.8 * len(data))
    valid_len = int(0.1 * len(data))
    test_len = len(data) - train_len - valid_len
    train_dataset, valid_dataset, test_dataset = random_split(data, [train_len, valid_len, test_len])
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = LeNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    warmup_steps = 10
    lr_lambda = lambda epoch: epoch / warmup_steps if epoch < warmup_steps else 1
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train_model(train_loader, valid_loader, optimizer, nn.CrossEntropyLoss(), scheduler, logger, num_epochs=25, device=device)
    model.check_test_accuracy(test_loader, device=device)

    logger.close()

