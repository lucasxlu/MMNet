import torch.nn.functional as F
import torch.nn as nn


def swish(x):
    return x * F.sigmoid(x)


class Swish(nn.Module):
    """
    implementation of Swish activation function
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, input):
        return input * F.sigmoid(input)
