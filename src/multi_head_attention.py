import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from scaled_dot_product import scaled_dot_product, diff_scaled_dot_product
from utils import total_params, plot_model_architecture


class MultiHeadAttention(nn.Module):
    def __init__(self, nheads: int = 4, dimension: int = 256):
        super(MultiHeadAttention, self).__init__()
        self.nheads = nheads
        self.dimension = dimension

        assert dimension % nheads == 0, "Dimension must be divisible by nheads"

        self.QKV = nn.Linear(
            in_features=self.dimension, out_features=3 * self.dimension
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            QKV = self.QKV(x)

            query, key, value = torch.chunk(input=QKV, chunks=3, dim=-1)

            query = query.view(
                query.size(0), query.size(1), self.nheads, self.dimension // self.nheads
            )
            key = key.view(
                key.size(0), key.size(1), self.nheads, self.dimension // self.nheads
            )
            value = value.view(
                value.size(0), value.size(1), self.nheads, self.dimension // self.nheads
            )

            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)

            attention = scaled_dot_product(query=query, key=key, value=value)

            attention = attention.view(
                attention.size(0),
                attention.size(2),
                attention.size(1) * attention.size(3),
            )
            return attention

        else:
            raise ValueError("Input must be a torch.Tensor".capitalize())

    @staticmethod
    def total_params(model):
        if isinstance(model, MultiHeadAttention):
            print("Total Parameters: ", total_params(model=model))
        else:
            raise ValueError("Input must be a MultiHeadAttention".capitalize())


class DifferentialMultiHeadAttention(nn.Module):
    def __init__(self, nheads: int = 4, dimension: int = 256, lam: float = 0.5):
        super(DifferentialMultiHeadAttention, self).__init__()
        self.nheads = nheads
        self.dimension = dimension
        self.lam = lam
        assert dimension % nheads == 0, "Dimension must be divisible by nheads"

        self.W_q = nn.Linear(self.dimension, 2 * self.dimension)
        self.W_k = nn.Linear(self.dimension, 2 * self.dimension)
        self.W_v = nn.Linear(self.dimension, self.dimension)

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")

        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)

        query = query.view(
            x.size(0), x.size(1), self.nheads, 2 * (self.dimension // self.nheads)
        )
        key = key.view(
            x.size(0), x.size(1), self.nheads, 2 * (self.dimension // self.nheads)
        )
        value = value.view(
            x.size(0), x.size(1), self.nheads, self.dimension // self.nheads
        )

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention = diff_scaled_dot_product(
            query=query, key=key, value=value, lam=self.lam
        )

        attention = attention.permute(0, 2, 1, 3).contiguous()
        attention = attention.view(x.size(0), x.size(1), self.dimension)

        return attention

    @staticmethod
    def total_params(model):
        if isinstance(model, DifferentialMultiHeadAttention):
            print("Total Parameters: ", total_params(model=model))
        else:
            raise ValueError("Input must be a DifferentialMultiHeadAttention instance.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test MultiHead and Differential Attention"
    )
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--use_diff", action="store_true")
    args = parser.parse_args()

    x = torch.randn(1, 256, args.dimension)

    if args.use_diff:
        model = DifferentialMultiHeadAttention(
            nheads=args.nheads, dimension=args.dimension, lam=args.lam
        )
    else:
        model = MultiHeadAttention(nheads=args.nheads, dimension=args.dimension)

    y = model(x)
    print(f"Output shape: {y.shape}")
