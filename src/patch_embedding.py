import os
import sys
import torch
import argparse
import warnings
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

        self.num_of_patches = (self.image_size // self.patch_size) ** 2
        self.in_channels = (self.dimension // self.image_size) * 2 * self.image_size

        self.projection = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.dimension,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=self.patch_size // self.patch_size,
            bias=self.bias,
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = self.projection(x)
            return x
        else:
            raise ValueError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    patch_embedding = PatchEmbedding(image_size=128, patch_size=4, dimension=512)
    images = torch.randn((16, 1024, 8, 8))
    print(patch_embedding(x=images).size())
