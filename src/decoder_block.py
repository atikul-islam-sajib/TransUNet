import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int = 256, out_channels: int = 128):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    pass
