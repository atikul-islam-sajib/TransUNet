import os
import sys
import torch
import torch.optim as optim

sys.path.append("./src/")

from loss.bce_loss import BCE
from transUNet import TransUNet
from loss.dice_loss import DiceLoss
from loss.focal_loss import FocalLoss
from loss.jaccard_loss import JaccardLoss
from loss.tversky_loss import TverskyLoss
from loss.mse_loss import MeanSquaredLoss
from loss.combined_loss import CombinedLoss
from utils import path_names, load_file, config_files


def load_dataloader():
    processed_path = path_names()["processed_data_path"]

    train_dataloader = os.path.join(processed_path, "train_dataloader.pkl")
    valid_dataloader = os.path.join(processed_path, "test_dataloader.pkl")

    train_dataloader = load_file(filename=train_dataloader)
    valid_dataloader = load_file(filename=valid_dataloader)

    return {"train_dataloader": train_dataloader, "valid_dataloader": valid_dataloader}


def helper(**kwargs):
    model: TransUNet = kwargs["model"]
    lr: float = kwargs["lr"]
    beta1: float = kwargs["beta1"]
    beta2: float = kwargs["beta2"]
    weight_decay: float = kwargs["weight_decay"]
    momentum: float = kwargs["momentum"]
    adam: bool = kwargs["adam"]
    SGD: bool = kwargs["SGD"]
    loss: str = kwargs["loss"]
    loss_smooth: float = kwargs["loss_smooth"]
    alpha_focal: float = kwargs["alpha_focal"]
    gamma_focal: float = kwargs["gamma_focal"]
    alpha_tversky: float = kwargs["alpha_tversky"]
    beta_tversky: float = kwargs["beta_tversky"]

    image_channels: int = config_files()["dataloader"]["image_channels"]
    image_size: int = config_files()["dataloader"]["image_size"]
    nheads: int = config_files()["TransUNet"]["nheads"]
    num_layers: int = config_files()["TransUNet"]["num_layers"]
    dim_feedforward: int = config_files()["TransUNet"]["dim_feedforward"]
    dropout: float = float(config_files()["TransUNet"]["dropout"])
    activation: str = config_files()["TransUNet"]["activation"]
    layer_norm_eps: float = float(config_files()["TransUNet"]["layer_norm_eps"])
    bias: bool = config_files()["TransUNet"]["bias"]

    if model is None:
        trans_unet = TransUNet(
            image_channels=image_channels,
            image_size=image_size,
            nheads=nheads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
        )
    elif isinstance(model, TransUNet):
        trans_unet = model
    else:
        raise ValueError("Invalid model type. Expected TransUNet.".capitalize())

    if adam:
        optimizer = optim.Adam(
            params=trans_unet.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
    elif SGD:
        optimizer = optim.SGD(
            params=trans_unet.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    if loss == "bce" or loss == "BCE":
        criterion = BCE(name="BCEWithLogitsLoss")
    elif loss == "dice":
        criterion = DiceLoss(smooth=loss_smooth)
    elif loss == "focal":
        criterion = FocalLoss(alpha=alpha_focal, gamma=gamma_focal)
    elif loss == "jaccard":
        criterion = JaccardLoss(smooth=loss_smooth)
    elif loss == "tversky":
        criterion = TverskyLoss(
            alpha=alpha_tversky, beta=beta_tversky, smooth=loss_smooth
        )
    elif loss == "mse" or loss == "MSE":
        criterion = MeanSquaredLoss(reduction="mean")
    else:
        raise ValueError(
            "Invalid loss function. Expected one of the following: bce, dice, focal, jaccard, tversky, mse"
        )

    return {
        "train_dataloader": load_dataloader()["train_dataloader"],
        "valid_dataloader": load_dataloader()["valid_dataloader"],
        "model": trans_unet,
        "optimizer": optimizer,
        "criterion": criterion,
    }


if __name__ == "__main__":
    adam, SGD = True, False
    loss = "bce"
    """
    "dice" | "focal" | "jaccard" | "tversky" | "BCE"
    """
    init = helper(
        model=None,
        lr=2e-4,
        beta1=0.9,
        beta2=0.999,
        weight_decay=1e-5,
        momentum=0.9,
        adam=True,
        SGD=False,
        loss="bce",
        loss_smooth=1e-6,
        alpha_focal=0.25,
        gamma_focal=2.0,
        alpha_tversky=0.5,
        beta_tversky=0.5,
    )

    assert init["train_dataloader"].__class__ == torch.utils.data.DataLoader
    assert init["valid_dataloader"].__class__ == torch.utils.data.DataLoader
    assert init["model"].__class__ == TransUNet
    if adam:
        assert init["optimizer"].__class__ == torch.optim.Adam
    elif SGD:
        assert init["optimizer"].__class__ == torch.optim.SGD

    if loss == "bce" or loss == "BCE":
        assert init["criterion"].__class__ == BCE
    elif loss == "dice":
        assert init["criterion"].__class__ == DiceLoss
    elif loss == "focal":
        assert init["criterion"].__class__ == FocalLoss
    elif loss == "tversky":
        assert init["criterion"].__class__ == TverskyLoss
    elif loss == "jaccard":
        assert init["criterion"].__class__ == JaccardLoss
    else:
        raise ValueError(
            "Invalid loss function. Expected one of: bce, dice, focal, jaccard, tversky.".capitalize()
        )
