import os
import sys
import math
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class LayerNormalization(nn.Module):
    def __init__(self, dimension: int = 512, layer_norm_eps: float = 1e-05):
        super(LayerNormalization, self).__init__()
        self.dimension = dimension
        self.layer_norm_eps = layer_norm_eps

        self.alpha = nn.Parameter(
            data=torch.ones((1, 1, self.dimension)), requires_grad=True
        )
        self.beta = nn.Parameter(
            data=torch.zeros((1, 1, self.dimension)), requires_grad=True
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor".capitalize())

        mean = torch.mean(input=x, dim=-1)
        variance = torch.var(input=x, dim=-1)

        mean = mean.unsqueeze(-1)
        variance = variance.unsqueeze(-1)

        normalized_x = (x - mean) / torch.sqrt(variance + self.layer_norm_eps)

        return self.alpha * normalized_x + self.beta


if __name__ == "__main__":
    layer_norm = LayerNormalization(dimension=512)
    images = torch.randn((1, 256, 512))
    print(layer_norm(x=images).size())
