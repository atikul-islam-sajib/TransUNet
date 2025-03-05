import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class MeanSquaredLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(MeanSquaredLoss, self).__init__()
        self.reduction = reduction

        self.criterion = nn.MSELoss(reduction=self.reduction)

    def forward(self, predicted: torch.Tensor, actual: torch.Tensor):
        if not isinstance(predicted, torch.Tensor) and not isinstance(
            actual, torch.Tensor
        ):
            raise TypeError(
                "Predicted and actual values must be of type torch.Tensor".capitalize()
            )

        predicted = torch.tanh(input=predicted)

        return self.criterion(predicted, actual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mean Squared Loss".title())

    args = parser.parse_args()

    predicted = torch.tensor([1.0, 2.0, 3.0])
    actual = torch.tensor([2.0, 3.0, 4.0])

    mse_loss = MeanSquaredLoss(reduction=args.reduction)

    assert type(mse_loss(predicted, actual)) == torch.Tensor
