import os
import sys
import torch
import argparse
import warnings
import torch.nn as nn

sys.path.append("./src/")

from utils import total_params, plot_model_architecture


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
        
    @staticmethod
    def total_parameters(model):
        if isinstance(model, PatchEmbedding):
            print("Total Parameters: ", total_params(model)) 
        else:
            raise ValueError("Input must be a PatchEmbedding".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch Embedding".title())
    parser.add_argument(
        "--image_size", type=int, default=128, help="Image size".capitalize()
    )
    parser.add_argument(
        "--patch_size", type=int, default=1, help="Patch size".capitalize()
    )
    parser.add_argument(
        "--dimension", type=int, default=1024, help="Dimension".capitalize()
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Enable bias for the convolution layer".capitalize(),
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the model architecture".capitalize(),
    )

    args = parser.parse_args()

    patch_embedding = PatchEmbedding(
        image_size=args.image_size,
        patch_size=args.patch_size // args.patch_size,
        dimension=args.dimension,
        bias=args.bias,
    )

    images = torch.randn((16, args.dimension, 8, 8))

    assert (patch_embedding(x=images).size()) == (16, 8 * 8, args.dimension)

    if args.display:
        plot_model_architecture(
            model=patch_embedding,
            input_data=images,
            model_name="Patch Embedding",
            format="pdf",
        )
