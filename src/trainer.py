import os
import sys
import torch
import argparse
import traceback
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("./src/")

from transUNet import TransUNet
from utils import (
    device_init,
    config_files,
    plot_images,
    path_names,
    dump_files,
    load_file,
    IoUScore,
)
from helper import helper
from loss.bce_loss import BCE
from transUNet import TransUNet
from loss.dice_loss import DiceLoss
from loss.focal_loss import FocalLoss
from loss.jaccard_loss import JaccardLoss
from loss.tversky_loss import TverskyLoss
from loss.mse_loss import MeanSquaredLoss
from loss.combined_loss import CombinedLoss


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
        optimizer: str = "AdamW",
        l1_regularization: bool = False,
        elastic_net_regularization: bool = False,
        verbose: bool = True,
        device: str = "cuda",
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
        self.optimizer = optimizer
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
            optimizer=self.optimizer,
            loss=self.loss_func,
            loss_smooth=self.loss_smooth,
            alpha_focal=self.alpha_focal,
            gamma_focal=self.gamma_focal,
            alpha_tversky=self.alpha_tversky,
            beta_tversky=self.beta_tversky,
        )

        self.IoU = IoUScore(threshold=0.5, smooth=1e-5)

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
        elif self.loss_func == "mse" or self.loss_func == "MSE":
            assert self.init["criterion"].__class__ == MeanSquaredLoss
        else:
            assert self.init["criterion"].__class__ == CombinedLoss

        self.loss = float("inf")
        self.history = {
            "train_loss": [],
            "valid_loss": [],
            "train_IoU": [],
            "valid_IoU": [],
        }

    def l1_regularizer(self, model: TransUNet = None):
        if (model is None) and (not isinstance(model, TransUNet)):
            raise ValueError("Invalid model. Expected TransUNet.".capitalize())

        return self.weight_decay * sum(
            torch.norm(params, 1) for params in model.parameters()
        )

    def elastic_net_regularizer(self, model: TransUNet = None):
        if (model is None) and (not isinstance(model, TransUNet)):
            raise ValueError("Invalid model. Expected TransUNet.".capitalize())

        l1_regularization = sum(torch.norm(params, 1) for params in model.parameters())
        l2_regularization = sum(torch.norm(params, 2) for params in model.parameters())

        return self.weight_decay * (l1_regularization + l2_regularization)

    def saved_checkpoints(self, **kwargs):
        valid_loss = kwargs["valid_loss"]
        train_loss = kwargs["train_loss"]
        train_epoch = kwargs["epoch"]

        best_model_path = path_names()["best_model_path"]
        train_model_path = path_names()["train_models_path"]

        if self.loss > train_loss:
            self.loss = train_loss
            torch.save(
                {
                    "epoch": train_epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                },
                os.path.join(best_model_path, "best_model.pth"),
            )

        torch.save(
            self.model.state_dict(),
            os.path.join(train_model_path, f"model{train_epoch}.pth"),
        )

    def display_progress(self, **kwargs):
        train_loss = kwargs["train_loss"]
        valid_loss = kwargs["valid_loss"]

        train_IoU = kwargs["train_IoU"]
        valid_IoU = kwargs["valid_IoU"]

        epochs_run = kwargs["epoch"]

        print(
            "Epoch: [{}/{}] - train_loss: {:.4f} - test_loss: {:.4f} - train_IoU: {:.4f} - test_IoU: {:.4f}".format(
                epochs_run, self.epochs, train_loss, valid_loss, train_IoU, valid_IoU
            )
        )

    def update_train(self, **kwargs):
        segmented = kwargs["segmented"]
        predicted = kwargs["predicted"]

        self.optimizer.zero_grad()

        train_loss = self.criterion(segmented, predicted)

        if self.l1_regularization:
            train_loss += self.l1_regularizer(model=self.model)
        elif self.elastic_net_regularization:
            train_loss += self.elastic_net_regularizer(model=self.model)

        train_loss.backward()
        self.optimizer.step()

        return {"train_loss": train_loss.item()}

    def plot_train_images(self, epoch: int = 1):
        images, masks = next(iter(self.valid_dataloader))
        original_images = images.to(self.device)
        original_masks = masks.to(self.device)

        predicted = self.model(original_images)

        plot_images(
            original_images=original_images,
            original_masks=original_masks,
            predicted_images=predicted,
            predicted=True,
            epoch=epoch,
        )

    def train(self):
        for epoch in tqdm(range(self.epochs), desc="Training TransUNet"):
            try:
                train_IoU = []
                valid_IoU = []
                train_loss = []
                valid_loss = []

                for index, (images, segmented) in enumerate(self.train_dataloader):
                    try:
                        images = images.to(self.device)
                        segmented = segmented.to(self.device)

                        predicted = self.model(images)

                        loss = self.update_train(
                            segmented=segmented, predicted=predicted
                        )
                        IoU = self.IoU(predicted=predicted, actual=segmented)

                        train_loss.append(loss["train_loss"])
                        train_IoU.append(IoU.item())

                    except Exception as e:
                        print(
                            f"Error in training loop (batch {index}, epoch {epoch+1}): {e}"
                        )
                        traceback.print_exc()

                for index, (images, segmented) in enumerate(self.valid_dataloader):
                    try:
                        images = images.to(self.device)
                        segmented = segmented.to(self.device)

                        predicted = self.model(images)
                        loss = self.criterion(segmented, predicted)
                        IoU = self.IoU(predicted=predicted, actual=segmented)

                        valid_loss.append(loss.item())
                        valid_IoU.append(IoU.item())

                    except Exception as e:
                        print(
                            f"Error in validation loop (batch {index}, epoch {epoch+1}): {e}"
                        )
                        traceback.print_exc()

                try:
                    self.display_progress(
                        train_loss=np.mean(train_loss),
                        valid_loss=np.mean(valid_loss),
                        train_IoU=np.mean(train_IoU),
                        valid_IoU=np.mean(valid_IoU),
                        epoch=epoch + 1,
                    )
                except Exception as e:
                    print(f"Error displaying progress in epoch {epoch+1}: {e}")
                    traceback.print_exc()

                try:
                    self.plot_train_images(epoch=epoch + 1)
                    self.saved_checkpoints(
                        train_loss=np.mean(train_loss),
                        valid_loss=np.mean(valid_loss),
                        epoch=epoch + 1,
                    )
                except Exception as e:
                    print(f"Error plotting images in epoch {epoch+1}: {e}")
                    traceback.print_exc()

                self.history["train_loss"].append(np.mean(train_loss))
                self.history["valid_loss"].append(np.mean(valid_loss))
                self.history["train_IoU"].append(np.mean(train_IoU))
                self.history["valid_IoU"].append(np.mean(valid_IoU))

            except KeyboardInterrupt:
                print("\nTraining interrupted manually. Saving progress...")
                break
            except Exception as e:
                print(f"Unexpected error in epoch {epoch+1}: {e}")
                traceback.print_exc()

        dump_files(
            value=self.history,
            filename=os.path.join(path_names()["metrics_path"], "history.pkl"),
        )

    @staticmethod
    def display_history():
        metrics_path = path_names()["metrics_path"]
        history_file = os.path.join(metrics_path, "history.pkl")

        if not os.path.exists(history_file):
            print("No history found.")
            return

        history = load_file(filename=history_file)

        if history is None or not isinstance(history, dict):
            print("Invalid or empty history file.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

        axes[0, 0].plot(history["train_loss"], label="Train Loss", color="blue")
        axes[0, 0].plot(history["valid_loss"], label="Validation Loss", color="red")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epochs")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()

        axes[0, 1].plot(history["train_IoU"], label="Train IoU", color="green")
        axes[0, 1].plot(history["valid_IoU"], label="Validation IoU", color="purple")
        axes[0, 1].set_title("IoU Score")
        axes[0, 1].set_xlabel("Epochs")
        axes[0, 1].set_ylabel("IoU Score")
        axes[0, 1].legend()

        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

        plt.tight_layout()
        save_path = os.path.join(metrics_path, "history.png")
        plt.savefig(save_path)
        plt.show()

        print(f"History saved as '{save_path}'.")


if __name__ == "__main__":
    trainer_config = config_files()["trainer"]

    epochs = trainer_config["epochs"]
    lr = trainer_config["lr"]

    optimizer = trainer_config["optimizer"]
    adam = optimizer["adam"]
    beta1 = adam["beta1"]
    beta2 = adam["beta2"]
    weight_decay = float(adam["weight_decay"])

    SGD = optimizer["SGD"]
    momentum = SGD["momentum"]
    SGD_weight_decay = float(SGD["weight_decay"])

    AdamW = optimizer["AdamW"]
    beta1 = AdamW["beta1"]
    beta2 = AdamW["beta2"]
    weight_decay = float(AdamW["weight_decay"])

    loss = trainer_config["loss"]
    loss_type = loss["type"]
    loss_smooth = float(loss["loss_smooth"])
    alpha_focal = float(loss["alpha_focal"])
    gamma_focal = float(loss["gamma_focal"])
    alpha_tversky = float(loss["alpha_tversky"])
    beta_tversky = float(loss["beta_tversky"])
    l1_regularization = trainer_config["l1_regularization"]
    elastic_net_regularization = trainer_config["elastic_net_regularization"]
    verbose = trainer_config["verbose"]
    device = trainer_config["device"]

    parser = argparse.ArgumentParser(
        description="Train the TransUNet model for the segmentation dataset.".title()
    )
    parser.add_argument(
        "--epochs", type=int, default=epochs, help="Number of epochs for training"
    )
    parser.add_argument(
        "--lr", type=float, default=lr, help="Learning rate for training"
    )
    parser.add_argument(
        "--beta1", type=float, default=beta1, help="Beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=beta2, help="Beta2 for Adam optimizer"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=weight_decay,
        help="Weight decay for Adam optimizer",
    )
    parser.add_argument(
        "--momentum", type=float, default=momentum, help="Momentum for SGD optimizer"
    )
    parser.add_argument(
        "--loss_smooth", type=str, default=loss_smooth, help="Loss smooth type"
    )
    parser.add_argument(
        "--alpha_focal", type=float, default=alpha_focal, help="Alpha for Focal loss"
    )
    parser.add_argument(
        "--gamma_focal", type=float, default=gamma_focal, help="Gamma for Focal loss"
    )
    parser.add_argument(
        "--alpha_tversky",
        type=float,
        default=alpha_tversky,
        help="Alpha for Tversky loss",
    )
    parser.add_argument(
        "--beta_tversky", type=float, default=beta_tversky, help="Beta for Tversky loss"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=optimizer,
        help="Select the optimizer",
        choices=["Adam", "AdamW", "SGD"],
    )
    parser.add_argument(
        "--l1_regularization", action="store_true", help="Use L1 regularization"
    )
    parser.add_argument(
        "--elastic_net_regularization",
        action="store_true",
        help="Use Elastic net regularization",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Display progress and save images"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="Device for training (cuda or cpu)"
    )
    parser.add_argument(
        "--loss_type", type=str, default=loss_type, help="Define the loss type"
    )

    args = parser.parse_args()

    trainer = Trainer(
        model=None,
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        loss_func=loss_type,
        loss_smooth=args.loss_smooth,
        alpha_focal=args.alpha_focal,
        gamma_focal=args.gamma_focal,
        alpha_tversky=args.alpha_tversky,
        beta_tversky=args.beta_tversky,
        optimizer=args.optimizer,
        l1_regularization=args.l1_regularization,
        elastic_net_regularization=args.elastic_net_regularization,
        verbose=args.verbose,
        device=args.device,
    )

    trainer.train()
    Trainer.display_history()
