import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import plot_model_architecture


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int = 128, out_channels: int = 2 * 128):
        super(EncoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = 1
        self.stride_size = 1
        self.padding_size = self.kernel_size // 2

        self.layers = list()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride_size * 2,
                padding=self.padding_size,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.out_channels),
        )

        self.in_channels = self.in_channels
        self.out_channels = self.out_channels

        for index in range(3):
            if index != 1:
                self.kernel_size = self.kernel_size // self.kernel_size
                self.stride_size = self.stride_size // self.stride_size
                self.padding_size = self.kernel_size // 2
            else:
                self.kernel_size = self.kernel_size * 3
                self.stride_size = self.stride_size * 2
                self.padding_size = self.kernel_size // self.kernel_size

            self.layers.append(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride_size,
                    padding=self.padding_size,
                    bias=False,
                )
            )
            self.layers.append(nn.BatchNorm2d(num_features=self.out_channels))

            self.in_channels = self.out_channels

        self.layers.append(nn.ReLU(inplace=True))

        self.encoder_block = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            x1 = self.conv1(x)
            x2 = self.encoder_block(x)
            x = x1 + x2
            return x
        else:
            raise ValueError("Input must be a torch.Tensor".capitalize())

    @staticmethod
    def total_params(model):
        if isinstance(model, EncoderBlock):
            return sum(p.numel() for p in model.parameters())
        else:
            raise ValueError("Input must be an EncoderBlock".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encoder Block for the TransUNet".title()
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=128,
        help="Input channels to encode".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=256,
        help="Output channels to decode".capitalize(),
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Display the model architecture.".capitalize(),
    )

    args = parser.parse_args()

    encoder_block = EncoderBlock(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
    )
    images = torch.randn((16, 128, 64, 64))
    assert (encoder_block(x=images).size()) == (16, 128 * 2, 64 // 2, 64 // 2)

    if args.display:
        plot_model_architecture(
            model=encoder_block,
            input_data=images,
            model_name="Encoder Block",
            format="pdf",
        )
