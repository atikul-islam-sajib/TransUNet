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
from decoder_block import DecoderBlock
from utils import total_params, plot_model_architecture


class TransUNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 256,
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

        self.decoder1 = DecoderBlock(
            in_channels=self.in_channels * 8, out_channels=self.in_channels * 4
        )
        self.decoder2 = DecoderBlock(
            in_channels=self.in_channels * 8, out_channels=self.in_channels * 2
        )
        self.decoder3 = DecoderBlock(
            in_channels=self.in_channels * 4, out_channels=self.in_channels
        )
        self.decoder4 = DecoderBlock(
            in_channels=self.in_channels * 2,
            out_channels=self.in_channels // self.in_channels,
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

            decoder1 = self.decoder1(bottleneck)
            decoder1 = torch.concat((decoder1, encoder2), dim=1)

            decoder2 = self.decoder2(decoder1)
            decoder2 = torch.concat((decoder2, encoder1), dim=1)

            decoder3 = self.decoder3(decoder2)
            decoder3 = torch.concat((decoder3, x), dim=1)

            output = self.decoder4(decoder3)

            return output
        else:
            raise ValueError("Input must be a torch.Tensor".capitalize())

    @staticmethod
    def total_parameters(model):
        if not isinstance(model, TransUNet):
            raise ValueError("Input must be a TransUNet".capitalize())

        print("Total parameters: ", total_params(model=model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TransUNet architecture for the segmentation".title()
    )
    parser.add_argument(
        "--image_channels",
        type=int,
        default=3,
        help="Number of channels in the input image".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Size of the input image".capitalize(),
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=4,
        help="Number of heads in the multi-head attention".capitalize(),
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="Number of transformer encoder layers".capitalize(),
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=512,
        help="Dimension of the feedforward network in the transformer encoder".capitalize(),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate in the transformer encoder".capitalize(),
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function in the transformer encoder".capitalize(),
    )
    parser.add_argument(
        "--layer_norm_eps",
        type=float,
        default=1e-05,
        help="Epsilon value for layer normalization in the transformer encoder".capitalize(),
    )
    parser.add_argument(
        "--bias",
        type=bool,
        default=False,
        help="Add bias to the transformer encoder".capitalize(),
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Display the model architecture".capitalize(),
    )

    args = parser.parse_args()

    model = TransUNet(
        image_channels=args.image_channels,
        image_size=args.image_size,
        nheads=args.nheads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        layer_norm_eps=args.layer_norm_eps,
        bias=args.bias,
    )

    images = torch.randn((1, args.image_channels, args.image_size, args.image_size))

    assert model(x=images).size() == (1, 1, args.image_size, args.image_size)

    if args.display:
        plot_model_architecture(
            model=model, input_data=images, model_name="TransUNet", format="pdf"
        )
