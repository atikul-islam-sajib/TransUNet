import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from multi_head_attention import MultiHeadAttention
from layer_normalization import LayerNormalization
from feed_forward_network import FeedForwardNeuralNetwork
from utils import total_params, plot_model_architecture


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-05,
        bias: bool = False,
    ):
        super(TransformerEncoderBlock, self).__init__()
        self.dimension = dimension
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias

        self.multi_head_attention = MultiHeadAttention(
            nheads=self.nheads, dimension=self.dimension
        )
        self.layer_norm1 = LayerNormalization(
            dimension=self.dimension, layer_norm_eps=self.layer_norm_eps
        )
        self.layer_norm2 = LayerNormalization(
            dimension=self.dimension, layer_norm_eps=self.layer_norm_eps
        )
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.dropout2 = nn.Dropout(p=self.dropout)

        self.feed_forward_network = FeedForwardNeuralNetwork(
            dimension=self.dimension,
            dim_feedforward=self.dim_feedforward,
            activation=self.activation,
            dropout=self.dropout,
            bias=self.bias,
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a torch instance".capitalize())

        residual = x

        x = self.multi_head_attention(x=x)
        x = self.dropout1(x)
        x = torch.add(x, residual)
        x = self.layer_norm1(x)

        residual = x

        x = self.feed_forward_network(x=x)
        x = self.dropout2(x)
        x = torch.add(x, residual)
        x = self.layer_norm2(x)

        return x

    @staticmethod
    def total_parameters(model):
        if isinstance(model, TransformerEncoderBlock):
            return total_params(model=model)
        else:
            raise ValueError("Input must be a TransformerEncoderBlock".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Encoder Block".title())
    parser.add_argument(
        "--dimension", type=int, default=512, help="Embedding dimension".capitalize()
    )
    parser.add_argument(
        "--nheads", type=int, default=4, help="Number of heads".capitalize()
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=1024,
        help="Feedforward dimension".capitalize(),
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate".capitalize()
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function".capitalize(),
        choices=["relu", "gelu", "selu", "leaky"],
    )
    parser.add_argument(
        "--layer_norm_eps",
        type=float,
        default=1e-05,
        help="Layer normalization epsilon".capitalize(),
    )
    parser.add_argument("--bias", type=bool, default=False, help="Bias".capitalize())
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Display the model architecture".capitalize(),
    )

    args = parser.parse_args()

    transfomer = TransformerEncoderBlock(
        dimension=args.dimension,
        nheads=args.nheads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        layer_norm_eps=args.layer_norm_eps,
        bias=args.bias,
    )

    images = torch.randn((1, 256, args.dimension))

    assert (transfomer(x=images).size()) == (1, 256, args.dimension)

    if args.display:
        plot_model_architecture(
            model=transfomer,
            input_data=images,
            model_name="TransformerEncoderBlock",
            format="pdf",
        )
