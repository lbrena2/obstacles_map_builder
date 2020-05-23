import torch
from operator import mul
from functools import reduce
from torchsummary import summary


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        batch_size = x.size(0)
        return x.reshape((batch_size,) + self.shape)


class NN(torch.nn.Module):
    def __init__(self, in_channels, out_shape):
        super(NN, self).__init__()

        out_channels = reduce(mul, out_shape, 1)

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size=3,
                            padding=1, stride=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ELU(),
            torch.nn.Conv2d(16, 32, kernel_size=3,
                            padding=1, stride=2),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, kernel_size=3,
                            padding=1),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, kernel_size=3,
                            padding=1),
            torch.nn.ELU(),
            Flatten(),
            torch.nn.Linear(640, out_channels),
            Reshape(out_shape),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    model = NN(3, (20, 20))
    summary(model, (3, 64, 80), device='cpu')
