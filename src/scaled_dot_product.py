import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


def scaled_dot_product(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    if (
        isinstance(query, torch.Tensor)
        and isinstance(value, torch.Tensor)
        and isinstance(value, torch.Tensor)
    ):
        key = key.transpose(-2, -1)
        logits = torch.matmul(input=query, other=key)
        dimension = key.size(-1)
        logits = torch.softmax(
            input=(logits / torch.sqrt(torch.tensor(dimension))), dim=-1
        )
        attention = torch.matmul(input=logits, other=value)

        return attention
    else:
        raise ValueError("Inputs must be torch.Tensor".capitalize())


if __name__ == "__main__":
    query = torch.randn(1, 4, 64, 64)
    key = torch.randn(1, 4, 64, 64)
    value = torch.randn(1, 4, 64, 64)

    attention = scaled_dot_product(query=query, key=key, value=value)
    print(attention.size())
