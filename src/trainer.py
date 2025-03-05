import os
import sys
import torch
import traceback
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("./src/")

from transUNet import TransUNet
from utils import device_init, plot_images, path_names, load_file
from helper import helper
from loss.bce_loss import BCE
from transUNet import TransUNet
from loss.dice_loss import DiceLoss
from loss.focal_loss import FocalLoss
from loss.jaccard_loss import JaccardLoss
from loss.tversky_loss import TverskyLoss

class IoUScore(nn.Module):
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-5):
        super(IoUScore, self).__init__()
        self.threshold = threshold
        self.smooth = smooth

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())
        pass

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

        self.loss = float("inf")
        self.history = {"train_loss": [], "valid_loss": []}

    def l1_regularization(self, model: TransUNet = None):
        if (model is None) and (not isinstance(model, TransUNet)):
            raise ValueError("Invalid model. Expected TransUNet.".capitalize())

        return self.weight_decay * sum(
            torch.norm(params, 1) for params in model.parameters()
        )

    def elastic_net_regularization(self, model: TransUNet = None):
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

        if self.loss > valid_loss:
            self.loss = valid_loss
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
        epochs_run = kwargs["epoch"]

        print(
            "Epoch: [{}/{}] - train_loss: {:.4f} - test_loss: {:.4f}".format(
                epochs_run, self.epochs, train_loss, valid_loss
            )
        )

    def update_train(self, **kwargs):
        segmented = kwargs["segmented"]
        predicted = kwargs["predicted"]

        self.optimizer.zero_grad()

        train_loss = self.criterion(segmented, predicted)
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
                        train_loss.append(loss["train_loss"])
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
                        valid_loss.append(loss.item())
                    except Exception as e:
                        print(
                            f"Error in validation loop (batch {index}, epoch {epoch+1}): {e}"
                        )
                        traceback.print_exc()

                try:
                    self.display_progress(
                        train_loss=np.mean(train_loss),
                        valid_loss=np.mean(valid_loss),
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

            except KeyboardInterrupt:
                print("\nTraining interrupted manually. Saving progress...")
                break
            except Exception as e:
                print(f"Unexpected error in epoch {epoch+1}: {e}")
                traceback.print_exc()

    @staticmethod
    def display_history():
        metrics_path = path_names()["metrics_path"]
        metrics_path = os.path.join(metrics_path, "history.pkl")
        history = load_file(filename=metrics_path)

        if metrics_path is not None:
            _, axes = plt.subplots(1, 1, figsize=(10, 10), sharex=True)

            axes[0, 0].plot(history["train_loss"], label="Train Loss")
            axes[0, 0].plot(history["valid_loss"], label="Test Loss")
            axes[0, 0].set_title("Loss")
            axes[0, 0].set_xlabel("Epochs")
            axes[0, 0].legend()

            axes[0, 0].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(metrics_path, "history.png"))
            plt.show()
            print("History saved as 'history.png' in the metrics folder".capitalize())
        else:
            print("No history found".capitalize())


if __name__ == "__main__":
    trainer = Trainer(
        model=None,
        epochs=50,
        lr=0.0002,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        momentum=0.95,
        loss_smooth="dice",
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

    trainer.train()
