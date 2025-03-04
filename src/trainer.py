import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")

from transUNet import TransUNet
from utils import device_init
from helper import helper
from loss.bce_loss import BCE
from transUNet import TransUNet
from loss.dice_loss import DiceLoss
from loss.focal_loss import FocalLoss
from loss.jaccard_loss import JaccardLoss
from loss.tversky_loss import TverskyLoss


class Trainer:
    def __init__(
        self,
        model: TransUNet = None,
        epochs: int = 200,
        lr: float = 2e-04,
        beta1: float = 0.5,
        beta2: float = 0.999,
        weight_decay: float = 1e-04,
        momentum: float = 0.85,
        loss_smooth: float = 1e-04,
        alpha_focal: float = 0.75,
        gamma_focal: float = 2,
        alpha_tversky: float = 0.75,
        beta_tversky: float = 0.5,
        adam: bool = True,
        SGD: bool = False,
        l1_regularization: bool = False,
        elastic_net_regularization: bool = False,
        verbose: bool = True,
        device: str = "cuad",
        loss_func: str = "bce",
    ):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.loss_smooth = loss_smooth
        self.alpha_focal = alpha_focal
        self.gamma_focal = gamma_focal
        self.alpha_tversky = alpha_tversky
        self.beta_tversky = beta_tversky
        self.adam = adam
        self.SGD = SGD
        self.l1_regularization = l1_regularization
        self.elastic_net_regularization = elastic_net_regularization
        self.verbose = verbose
        self.device = device
        self.loss_func = loss_func

        self.device = device_init(device=self.device)

        self.init = helper(
            model=self.model,
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            adam=self.adam,
            SGD=self.SGD,
            loss=self.loss_func,
            loss_smooth=self.loss_smooth,
            alpha_focal=self.alpha_focal,
            gamma_focal=self.gamma_focal,
            alpha_tversky=self.alpha_tversky,
            beta_tversky=self.beta_tversky,
        )

        self.train_dataloader = self.init["train_dataloader"]
        self.valid_dataloader = self.init["valid_dataloader"]

        self.model = self.init["model"]

        self.optimizer = self.init["optimizer"]

        self.criterion = self.init["criterion"]

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        assert self.init["train_dataloader"].__class__ == torch.utils.data.DataLoader
        assert self.init["valid_dataloader"].__class__ == torch.utils.data.DataLoader
        assert self.init["model"].__class__ == TransUNet
        if self.adam:
            assert self.init["optimizer"].__class__ == torch.optim.Adam
        elif self.SGD:
            assert self.init["optimizer"].__class__ == torch.optim.SGD

        if self.loss_func == "bce" or self.criterion == "BCE":
            assert self.init["criterion"].__class__ == BCE
        elif self.loss_func == "dice":
            assert self.init["criterion"].__class__ == DiceLoss
        elif self.loss_func == "focal":
            assert self.init["criterion"].__class__ == FocalLoss
        elif self.loss_func == "tversky":
            assert self.init["criterion"].__class__ == TverskyLoss
        elif self.loss_func == "jaccard":
            assert self.init["criterion"].__class__ == JaccardLoss
        else:
            raise ValueError(
                "Invalid loss function. Expected one of: bce, dice, focal, jaccard, tversky.".capitalize()
            )

    def l1_regularization(self, model: TransUNet = None):
        pass

    def elastic_net_regularization(self, model: TransUNet = None):
        pass

    def saved_checkpoints(self, **kwargs):
        pass

    def display_progress(self, **kwargs):
        pass

    def update_train(self, **kwargs):
        pass

    def train(self):
        pass


if __name__ == "__main__":
    trainer = Trainer(
        model=None,
        epochs=2,
        lr=0.001,
        beta1=0.5,
        beta2=0.999,
        weight_decay=0.0,
        momentum=0.95,
        loss_smooth="bce",
        alpha_focal=0.75,
        gamma_focal=2.0,
        alpha_tversky=0.75,
        beta_tversky=0.50,
        adam=True,
        SGD=False,
        l1_regularization=False,
        elastic_net_regularization=False,
        verbose=True,
        device="cuda",
    )
