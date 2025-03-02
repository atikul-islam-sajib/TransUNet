import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, reduction: str = "mean"):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, predicted: torch.Tensor, actual: torch.Tensor):
        if not isinstance(predicted, torch.Tensor) or not isinstance(
            actual, torch.Tensor
        ):
            raise ValueError("Inputs must be torch.Tensor")

        predicted = torch.sigmoid(predicted)

        predicted = predicted.view(predicted.size(0), -1)
        actual = actual.view(actual.size(0), -1)

        TP = (predicted * actual).sum(dim=1)
        FP = (predicted * (1 - actual)).sum(dim=1)
        FN = ((1 - predicted) * actual).sum(dim=1)

        epsilon = 1e-8
        tversky_index = TP / (TP + self.alpha * FP + self.beta * FN + epsilon)

        tversky_loss = 1 - tversky_index

        if self.reduction == "mean":
            return tversky_loss.mean()
        elif self.reduction == "sum":
            return tversky_loss.sum()
        else:
            return tversky_loss


if __name__ == "__main__":
    predicted = torch.randn((64, 1, 128, 128))
    actual = torch.randint(0, 2, (64, 1, 128, 128)).float()

    loss_func = TverskyLoss(alpha=0.7, beta=0.3, reduction="mean")
    print(loss_func(actual, predicted))
