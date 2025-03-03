import os
import sys
import math
import torch
import argparse
import warnings
import torch.nn as nn

sys.path.append("./src/")

from ViT import ViT
from encoder_block import EncoderBlock


class TransUNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 128,
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
        self.nheads = nheads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias

        warnings.warn(
            "Output channel configuration is determined based on image size:\n"
            "  - If image_size > 256, out_channels = (image_channels - 1) ** 8\n"
            "  - If image_size > 128, out_channels = (image_channels - 1) ** 6\n"
            "  - Otherwise, out_channels = (image_channels - 1) ** 5"
        )

        if self.image_size > math.pow(2, 8):
            self.out_channels = (self.image_channels - 1) ** 8
        elif self.image_size > math.pow(2, 7):
            self.out_channels = (self.image_channels - 1) ** 6
        else:
            self.out_channels = (self.image_channels - 1) ** 5

        self.kernel_size = (self.image_channels * 2) + 1
        self.stride_size = (self.kernel_size // self.kernel_size) + 1
        self.padding_size = self.kernel_size // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.image_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride_size,
                padding=self.padding_size,
                bias=self.bias,
            ),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(inplace=True),
        )

        self.in_channels = self.out_channels

        self.encoder1 = EncoderBlock(
            in_channels=self.in_channels, out_channels=self.in_channels * 2
        )
        self.encoder2 = EncoderBlock(
            in_channels=self.in_channels * 2, out_channels=self.in_channels * 4
        )
        self.encoder3 = EncoderBlock(
            in_channels=self.in_channels * 4, out_channels=self.in_channels * 8
        )

        self.vit = ViT(
            image_size=self.image_size,
            dimension=self.in_channels * 8,
            nheads=self.nheads,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            layer_norm_eps=self.layer_norm_eps,
            bias=self.bias,
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = self.conv1(x)

            encoder1 = self.encoder1(x)
            encoder2 = self.encoder2(encoder1)
            encoder3 = self.encoder3(encoder2)

            bottleneck = self.vit(x=encoder3)

            return (encoder3, bottleneck)
        else:
            raise ValueError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    model = TransUNet(
        image_channels=3,
        image_size=256,
        nheads=4,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-05,
        bias=False,
    )

    images = torch.randn((1, 3, 256, 256))
    encoder, bottleneck = model(x=images)

    print("Encoder output size:", encoder.size())
    print("Bottleneck output size:", bottleneck.size())
