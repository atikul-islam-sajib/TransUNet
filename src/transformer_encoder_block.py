import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from multi_head_attention import MultiHeadAttention
from layer_normalization import LayerNormalization
from feed_forward_network import FeedForwardNeuralNetwork


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


if __name__ == "__main__":
    transfomer = TransformerEncoderBlock(
        dimension=512,
        nheads=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-05,
        bias=False,
    )

    images = torch.randn((1, 256, 512))
    print(transfomer(x=images).size())
