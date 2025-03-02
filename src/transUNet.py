import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class TransUNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 128,
        patch_size: int = 16,
        dimension: int = 512,
        nheads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-05,
        bias: bool = False,
    ):
        super(TransUNet, self).__init__()

        self.image_channels = image_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.dimension = dimension
        self.nheads = nheads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias
