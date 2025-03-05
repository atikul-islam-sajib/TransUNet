import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import total_params, plot_model_architecture


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
            self.activation_func = nn.GELU()
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

    @staticmethod
    def total_params(model):
        if isinstance(model, FeedForwardNeuralNetwork):
            return total_params(model)
        else:
            raise ValueError("Input must be a FeedForwardNeuralNetwork".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feed Forward Neural Network".title())
    parser.add_argument(
        "--dimension", type=int, default=512, help="Dimension of the input data"
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=1024,
        help="Dimension of the feedforward layers",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function",
        choices=["relu", "gelu", "selu", "leaky"],
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Display the network".capitalize(),
    )
    args = parser.parse_args()
    network = FeedForwardNeuralNetwork(
        dimension=args.dimension,
        dim_feedforward=args.dim_feedforward,
        activation=args.activation,
        dropout=args.dropout,
    )
    images = torch.randn((1, 256, args.dimension))
    assert (network(x=images).size()) == (1, 256, args.dimension)

    if args.display:
        plot_model_architecture(
            model=network, input_data=images, model_name="FFNN", format="pdf"
        )
