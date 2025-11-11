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

    key = key.transpose(-1, -2)

    logits = torch.matmul(query, key)
    dimension = key.size(-1)

    logits = logits / torch.sqrt(
        torch.tensor(float(dimension), dtype=query.dtype, device=query.device)
    )
    logits = torch.softmax(logits, dim=-1)

    attention = torch.matmul(logits, value)

    return attention


import torch


def scaled_dot_product(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    """Standard scaled dot-product attention"""
    if not all(isinstance(t, torch.Tensor) for t in [query, key, value]):
        raise ValueError("Inputs must be torch.Tensor")

    key = key.transpose(-1, -2)
    d = key.size(-1)

    logits = torch.matmul(query, key) / torch.sqrt(
        torch.tensor(float(d), dtype=query.dtype, device=query.device)
    )
    attn = torch.softmax(logits, dim=-1)
    output = torch.matmul(attn, value)
    return output


def diff_scaled_dot_product(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, lam: float = 0.5
):
    if not all(isinstance(t, torch.Tensor) for t in [query, key, value]):
        raise ValueError("Inputs must be torch.Tensor")

    b, h, n, d2 = query.shape  # [batch, heads, seq, 2*d]
    d = d2 // 2

    Q1, Q2 = torch.split(query, d, dim=-1)
    K1, K2 = torch.split(key, d, dim=-1)

    K1 = K1.transpose(-1, -2)
    K2 = K2.transpose(-1, -2)

    scale = 1.0 / torch.sqrt(
        torch.tensor(float(d), dtype=query.dtype, device=query.device)
    )
    A1 = torch.matmul(Q1, K1) * scale
    A2 = torch.matmul(Q2, K2) * scale

    attn = torch.softmax(A1, dim=-1) - lam * torch.softmax(A2, dim=-1)
    output = torch.matmul(attn, value)
    return output


if __name__ == "__main__":
    query = torch.randn(1, 4, 256, 128)
    key = torch.randn(1, 4, 256, 128)
    value = torch.randn(1, 4, 256, 128)

    attention = scaled_dot_product(query, key, value)

    assert (attention.size()) == query.size() == key.size() == value.size()
