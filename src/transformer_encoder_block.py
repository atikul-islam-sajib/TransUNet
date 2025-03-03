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
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-05,
        bias: bool = False,
    ):
        super(TransformerEncoderBlock, self).__init__()
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a torch instance".capitalize())
        pass


if __name__ == "__main__":
    pass
