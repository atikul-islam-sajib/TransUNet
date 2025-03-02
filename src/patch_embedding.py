import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        dimension: int = 512,
        bias: bool = False,
    ):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dimension = dimension
        self.bias = bias

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError("Input must be a torch.Tensor".capitalize())
