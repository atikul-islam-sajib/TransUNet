import os
import sys
import torch
import argparse

sys.path.append("./src/")


def scaled_dot_product(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    if not (
        isinstance(query, torch.Tensor)
        and isinstance(key, torch.Tensor)
        and isinstance(value, torch.Tensor)
    ):
        raise ValueError("Inputs must be torch.Tensor")

    key = key.transpose(-2, -1)
    logits = torch.matmul(query, key)
    dimension = key.size(-1)

    logits = logits / torch.sqrt(
        torch.tensor(float(dimension), dtype=query.dtype, device=query.device)
    )
    logits = torch.softmax(logits, dim=-1)

    attention = torch.matmul(logits, value)

    return attention


if __name__ == "__main__":
    query = torch.randn(1, 4, 64, 64)
    key = torch.randn(1, 4, 64, 64)
    value = torch.randn(1, 4, 64, 64)

    attention = scaled_dot_product(query, key, value)
    print(attention.size())
