import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from loss.bce_loss import BCE
from loss.dice_loss import DiceLoss
from loss.focal_loss import FocalLoss
from loss.mse_loss import MeanSquaredLoss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1e-5,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
        combined_type: str = "focal + bce",
        weights: dict = None,
    ):

        super(CombinedLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.combined_type = combined_type.lower().replace(" ", "")  # Normalize input

        # Initialize loss functions
        self.loss_functions = {
            "bce": BCE(reduction=self.reduction),
            "dice": DiceLoss(reduction=self.reduction, smooth=self.smooth),
            "mse": MeanSquaredLoss(reduction=self.reduction),
            "focal": FocalLoss(
                alpha=self.alpha, gamma=self.gamma, reduction=self.reduction
            ),
        }

        # Default weights if none provided
        self.weights = weights or {key: 1.0 for key in self.loss_functions.keys()}

    def forward(self, inputs, targets):
        loss = 0.0
        for loss_name, loss_fn in self.loss_functions.items():
            if loss_name in self.combined_type or self.combined_type == "all":
                loss += self.weights.get(loss_name, 1.0) * loss_fn(inputs, targets)

        return loss
