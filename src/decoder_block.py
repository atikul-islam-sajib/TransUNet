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

        self.kernel_size = 1
        self.stride_size = 1
        self.padding_size = 1

        self.layers = []

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        for _ in range(2):
            self.layers.append(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size * 3,
                    stride=self.stride_size,
                    padding=self.padding_size,
                    bias=False,
                )
            )
            self.layers.append(nn.BatchNorm2d(num_features=self.out_channels))
            self.layers.append(nn.ReLU(inplace=True))

            self.in_channels = self.out_channels

        self.decoder_network = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())

        x = self.upsample(x)
        x = self.decoder_network(x)

        return x


if __name__ == "__main__":
    decoder_block = DecoderBlock(in_channels=256, out_channels=256 // 2)
    print(decoder_block)
    images = torch.randn((1, 256, 8, 8))
    print(decoder_block(x=images).size())
