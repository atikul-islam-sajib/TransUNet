import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        dim_feedforward: int = 1024,
        activation: str = "relu",
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.dimension = dimension
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.dropout = dropout
        self.bias = bias

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

        self.layers = list()

        for index in range(2):
            self.layers.append(
                nn.Linear(
                    in_features=self.dimension,
                    out_features=self.dim_feedforward,
                    bias=self.bias,
                )
            )
            self.dimension = self.dim_feedforward
            self.dim_feedforward = dimension

            if index == 0:
                self.layers.append(self.activation_func)
                self.layers.append(nn.Dropout(p=self.dropout))

        self.network = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor".capitalize())

        x = self.network(x)

        return x


if __name__ == "__main__":
    network = FeedForwardNeuralNetwork(dimension=512, dim_feedforward=1024)
    images = torch.randn((1, 256, 512))
    print(network(x=images).size())
