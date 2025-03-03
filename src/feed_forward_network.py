import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(
        self,
        dim_feedforward: int = 1024,
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.dropout = dropout

        if self.activation == "relu":
            self.activation_func = nn.ReLU(inplace=True)
        elif self.activation == "leaky":
            self.activation_func = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif self.activation == "gelu":
            self.activation_func = nn.GELU(inplace=True)
        elif self.activation == "selu":
            self.activation_func = nn.SELU(inplace=True)
        else:
            raise ValueError("Unsupported activation function".capitalize())

    def forward(self, x: torch.Tensor):
        if not isinstance(x, nn.Tensor):
            raise ValueError("Input must be a torch.Tensor".capitalize())
        pass


if __name__ == "__main__":
    pass
