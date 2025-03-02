import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class BCE(nn.Module):

    def __init__(self, name: str = "BCEWithLogitsLoss", reduction: str = "mean"):
        super(BCE, self).__init__()
        self.name = name
        self.reduction = reduction

        if self.name == "BCEWithLogitsLoss":
            self.loss_func = nn.BCEWithLogitsLoss(reduction=self.reduction)
        elif self.name == "BCELoss":
            self.loss_func = nn.BCELoss(reduction=self.reduction)

    def forward(self, predicted: torch.Tensor, actual: torch.Tensor):
        if isinstance(predicted, torch.Tensor) and isinstance(actual, torch.Tensor):
            return self.loss_func(predicted, actual)
        else:
            raise ValueError("Input must be torch.Tensor".capitalize())


if __name__ == "__main__":
    predicted = torch.Tensor([3.0, 2.0, -1.0, -3.0, 5.0])
    actual = torch.Tensor([1.0, 1.0, 0.0, 0.0, 1.0])

    loss = BCE(name="BCEWithLogitsLoss", reduction="mean")
    assert type(loss(predicted, actual)) == torch.Tensor
