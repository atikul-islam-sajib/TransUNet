import os
import sys
import math
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm

sys.path.append("./src/")

from patch_embedding import PatchEmbedding
from transformer_encoder_block import TransformerEncoderBlock
from utils import total_params, plot_model_architecture


class ViT(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        dimension: int = 512,
        nheads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-05,
        bias: bool = False,
        use_diff_attn: bool = False,  
        lam: float = 0.5,              
    ):
        super(ViT, self).__init__()
        self.image_size = image_size
        self.dimension = dimension
        self.nheads = nheads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias
        self.use_diff_attn = use_diff_attn  
        self.lam = lam                      

        self.patch_embedding = PatchEmbedding(
            image_size=self.image_size,
            patch_size=self.image_size // self.image_size,
            dimension=self.dimension,
            bias=self.bias,
        )

        self.transformer = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    dimension=self.dimension,
                    nheads=self.nheads,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                    activation=self.activation,
                    layer_norm_eps=self.layer_norm_eps,
                    bias=self.bias,
                    use_diff_attn=self.use_diff_attn,  
                    lam=self.lam,                      
                )
                for _ in tqdm(range(self.num_layers))
            ]
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor".capitalize())

        x = self.patch_embedding(x)

        for transformer in self.transformer:
            x = transformer(x)

        x = x.view(
            x.size(0), x.size(-1), int(math.sqrt(x.size(1))), int(math.sqrt(x.size(1)))
        )

        return x

    @staticmethod
    def total_parameters(model):
        if isinstance(model, ViT):
            print("Total Parameters: ", total_params(model))
        else:
            raise ValueError("Input must be a ViT".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vision Transformer, aka as ViT".title()
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", choices=["relu", "gelu", "silu"], default="relu")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-05)
    parser.add_argument("--bias", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--use_diff_attn", action="store_true")  
    parser.add_argument("--lam", type=float, default=0.5)       

    args = parser.parse_args()

    vit = ViT(
        image_size=args.image_size,
        dimension=args.dimension,
        nheads=args.nheads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        layer_norm_eps=args.layer_norm_eps,
        bias=args.bias,
        use_diff_attn=args.use_diff_attn,  
        lam=args.lam,                      
    )

    images = torch.randn((16, args.dimension, 16, 16))
    assert (vit(x=images).size()) == (16, args.dimension, 16, 16)

    if args.display:
        plot_model_architecture(
            model=vit,
            input_data=images,
            model_name="Vision Transformer (ViT)",
            format="pdf",
        )
