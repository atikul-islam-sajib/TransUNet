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
        patch_size: int = 1,
        dimension: int = 1024,
        bias: bool = False,
    ):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dimension = dimension
        self.bias = bias
        self.constant = 16

        assert (
            self.image_size % self.patch_size == 0
        ), "Image size must be divisible by the patch size.".title()

        warnings.warn(
            "The encoder block extracts features, which may reduce the effective image size. "
            "To mitigate this, we set the patch size to 1."
        )

        self.num_of_patches = (
            (self.image_size // self.constant) // self.patch_size
        ) ** 2
        self.in_channels, self.out_channels = self.dimension, self.dimension

        self.projection = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=self.bias,
        )

        self.postitonal_embedding = nn.Parameter(
            data=torch.randn((1, self.num_of_patches, self.dimension)),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = self.projection(x)
            x = x.view(x.size(0), x.size(-1) * x.size(-2), x.size(1))
            x = self.postitonal_embedding + x
            return x
        else:
            raise ValueError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    patch_embedding = PatchEmbedding(image_size=128, patch_size=1, dimension=1024)
    images = torch.randn((16, 1024, 8, 8))
    print(patch_embedding(x=images).size())
