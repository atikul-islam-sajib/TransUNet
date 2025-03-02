import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class DiceLoss(nn.Module):
    def __init__(self, reduction: str = "mean", smooth: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, predicted: torch.Tensor, actual: torch.Tensor):
        if not isinstance(predicted, torch.Tensor) or not isinstance(
            actual, torch.Tensor
        ):
            raise ValueError("Inputs must be torch.Tensor")

        predicted = torch.sigmoid(predicted)

        predicted = predicted.view(predicted.shape[0], -1)
        actual = actual.view(actual.shape[0], -1)

        intersection = (predicted * actual).sum(dim=1)
        union = predicted.sum(dim=1) + actual.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        loss = 1 - dice_score

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dice Loss".title())
    parser.add_argument(
        "--reduction", type=str, help="Reduction method", default="mean"
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=1e-5,
        help="Smooth valud of the Dice Loss".capitalize(),
    )
    args = parser.parse_args()

    predicted = torch.randn((64, 1, 128, 128))
    actual = torch.randint(0, 2, (64, 1, 128, 128)).float()

    loss_func = DiceLoss(reduction=args.reduction, smooth=args.smooth)
    assert type(loss_func(predicted, actual)) == torch.Tensor
