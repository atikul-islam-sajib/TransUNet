import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from patch_embedding import PatchEmbedding
from transformer_encoder_block import TransformerEncoderBlock


class ViT(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        nheads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-05,
        bias: bool = False,
    ):
        super(ViT, self).__init__()
        self.dimension = dimension
        self.nheads = nheads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor".capitalize())
        pass


if __name__ == "__main__":
    vit = ViT(
        dimension=512,
        nheads=4,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-05,
        bias=False,
    )
