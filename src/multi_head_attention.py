import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from scaled_dot_product import scaled_dot_product
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Head Attention".title())
    parser.add_argument(
        "--nheads", type=int, default=4, help="Number of heads".capitalize()
    )
    parser.add_argument(
        "--dimension", type=int, default=512, help="Dimension".capitalize()
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the model architecture".capitalize(),
    )

    args = parser.parse_args()

    attention = MultiHeadAttention(nheads=args.nheads, dimension=args.dimension)

    images = torch.randn(1, 256, args.dimension)

    assert (attention(x=images).size()) == (1, 256, args.dimension)

    if args.display:
        plot_model_architecture(
            model=attention,
            input_data=images,
            model_name="MultiHeadAttention",
            format="pdf",
        )
