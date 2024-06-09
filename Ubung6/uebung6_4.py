import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

def init_cnn(module):
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

class LeNet(d2l.Classifier):
    """The LeNet-5 model."""
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Conv2d(1,6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6,16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(400,120), nn.Sigmoid(),
            nn.Linear(120,84), nn.Sigmoid(),
            nn.Linear(84,num_classes))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Add this line

@d2l.add_to_class(d2l.Classifier)
def layer_summary(self, X_shape):
    X = torch.randn(*X_shape).to(self.device)  # Move the input tensor to the same device as the model
    for layer in self.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set the device
    print("Device: " + str(device))

    model = LeNet()
    model.to(device)  # Move the model to the device
    model.layer_summary((1, 1, 28, 28))

    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=128)

    model = LeNet(lr=0.1)
    model.to(device)  # Move the model to the device

    # Move the data to the device before applying the initialization
    model.apply_init([next(iter(data.get_dataloader(True)))[0].to(device)], init_cnn)

    trainer.fit(model, data)

    plt.show()
