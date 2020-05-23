#!/usr/bin/python

"""Function that creates the pytorch model."""

import torch
from torchsummary import summary


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batchsize = x.shape[0]
        return x.view(batchsize, -1)


class NN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NN, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=8,
                      kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=8, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=2, padding=2),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(2),
            Flatten(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=640, out_features=512),
            torch.nn.ELU(),
            torch.nn.Linear(in_features=512, out_features=out_channels),
            torch.nn.Sigmoid()
        )

    def summary(self, *args, **kwargs):
        summary(self, *args, **kwargs)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    model = NN(3, 65 * 5)
    model.summary((3, 64, 80), device='cpu')