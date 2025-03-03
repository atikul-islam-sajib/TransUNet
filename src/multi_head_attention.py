import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


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
            key = key.permute(0, 2, 3, 1)
            value = value.permute(0, 2, 1, 3)

            return query
        else:
            raise ValueError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    attention = MultiHeadAttention()
    images = torch.randn(1, 64, 256)
    print(attention(x=images).size())
