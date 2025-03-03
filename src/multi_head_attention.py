import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class MultiHeadAttention(nn.Module):
    def __init__(self, nheads: int = 4, dimension: int = 256):
        super(MultiHeadAttention, self).__init__()
        self.nheads = nheads
        self.dimension = dimension

        assert dimension % nheads == 0, "Dimension must be divisible by nheads"

        self.QKV = nn.Linear(
            in_features=self.dimension, out_features=3 * self.dimension
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    pass
