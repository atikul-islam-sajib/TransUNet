import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predicted: torch.Tensor, actual: torch.Tensor):
        if not isinstance(predicted, torch.Tensor) or not isinstance(
            actual, torch.Tensor
        ):
            raise ValueError("Inputs must be torch.Tensor")

        bce_loss = self.bce_loss(predicted, actual)

        with torch.no_grad():
            pt = torch.exp(-bce_loss)

        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dice Loss".title())
    parser.add_argument(
        "--reduction", type=str, help="Reduction method", default="mean"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="alpha value of the Dice Loss".capitalize(),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="gamma value of the Dice Loss".capitalize(),
    )
    args = parser.parse_args()

    predicted = torch.randn((64, 1, 128, 128))
    actual = torch.randint(0, 2, (64, 1, 128, 128)).float()

    loss_func = FocalLoss(alpha=args.alpha, gamma=args.gamma, reduction="mean")

    assert type(loss_func(predicted, actual)) == torch.Tensor
