import torch
from torch.nn import Module


class PlainMeanLoss(Module):
    def __init__(self):
        """
        A simple mean loss that averages the input tensor.
        """
        super().__init__()

    def forward(self, x):
        return torch.mean(x)
