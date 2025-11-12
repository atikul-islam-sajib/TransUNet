import os
import sys
import torch
import argparse
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("./src/")

from transUNet import TransUNet
from utils import (
    config_files,
    device_init,
    plot_images,
    load_file,
    path_names,
    IoUScore,
)
from loss.dice_loss import DiceLoss


class Tester:
    def __init__(self, dataset: str = "test", device: str = "cuda"):
        self.dataset = dataset
        self.device = device
        self.device = device_init(device=self.device)

        # Metrics
        self.iou_metric = IoUScore(threshold=0.5, smooth=1e-5)
        self.dice_metric = DiceLoss(smooth=1e-5)
        self.eps = 1e-7  # avoid div-by-zero

    def load_dataloader(self):
        processed_path = path_names()["processed_data_path"]
        train_dataloader_path = os.path.join(processed_path, "train_dataloader.pkl")
        valid_dataloader_path = os.path.join(processed_path, "test_dataloader.pkl")

        train_dataloader = load_file(filename=train_dataloader_path)
        valid_dataloader = load_file(filename=valid_dataloader_path)

        return {
            "train_dataloader": train_dataloader,
            "valid_dataloader": valid_dataloader,
        }

    def compute_additional_metrics(self, preds, targets):
        """Compute IoU, Dice, Precision, Recall, and F1-Score"""

        preds = (preds > 0.5).float()
        targets = (targets > 0.5).float()

        preds_flat = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (preds_flat * targets_flat).sum(dim=1)
        union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2 * intersection + self.eps) / (union + self.eps)
        iou = intersection / (union - intersection + self.eps)

        tp = (preds_flat * targets_flat).sum(dim=1)
        fp = (preds_flat * (1 - targets_flat)).sum(dim=1)
        fn = ((1 - preds_flat) * targets_flat).sum(dim=1)

        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)

        return {
            "iou": iou.mean().item(),
            "dice": dice.mean().item(),
            "precision": precision.mean().item(),
            "recall": recall.mean().item(),
            "f1": f1.mean().item(),
        }

    def display_images(self):
        try:
            dataset = self.load_dataloader()
            if self.dataset not in ["train", "test"]:
                raise ValueError(
                    "Invalid dataset. Please choose between 'train' or 'test'."
                )

            dataloader = (
                dataset["train_dataloader"]
                if self.dataset == "train"
                else dataset["valid_dataloader"]
            )

            best_model_path = os.path.join(
                path_names()["best_model_path"], "best_model.pth"
            )
            checkpoint = torch.load(best_model_path, map_location=self.device)
            model_state = checkpoint["model_state_dict"]

            model = TransUNet(
                image_channels=config_files()["dataloader"]["image_channels"],
                image_size=config_files()["dataloader"]["image_size"],
                nheads=config_files()["TransUNet"]["nheads"],
                num_layers=config_files()["TransUNet"]["num_layers"],
                dim_feedforward=config_files()["TransUNet"]["dim_feedforward"],
                dropout=float(config_files()["TransUNet"]["dropout"]),
                activation=config_files()["TransUNet"]["activation"],
                layer_norm_eps=float(config_files()["TransUNet"]["layer_norm_eps"]),
                bias=config_files()["TransUNet"]["bias"],
                use_diff_attn=config_files()["trainer"].get("use_diff_attn", False),
                lam=float(config_files()["trainer"].get("lam", 0.5)),
            )

            model.to(self.device)
            model.load_state_dict(model_state)
            model.eval()

            metrics = {"iou": [], "dice": [], "precision": [], "recall": [], "f1": []}

            with torch.no_grad():
                for index, (images, masks) in enumerate(tqdm(dataloader, desc="Testing")):
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    preds = model(images)

                    plot_images(
                        original_images=images,
                        original_masks=masks,
                        predicted_images=preds,
                        predicted=True,
                        epoch=index + 1,
                        testing=True,
                    )

                    batch_metrics = self.compute_additional_metrics(preds, masks)
                    for k in metrics.keys():
                        metrics[k].append(batch_metrics[k])

            mean_metrics = {k: np.mean(v) for k, v in metrics.items()}
            print("\n--- Overall Segmentation Performance ---")
            print(f"Mean IoU: {mean_metrics['iou']:.4f}")
            print(f"Mean Dice: {mean_metrics['dice']:.4f}")
            print(f"Mean Precision: {mean_metrics['precision']:.4f}")
            print(f"Mean Recall: {mean_metrics['recall']:.4f}")
            print(f"Mean F1-Score: {mean_metrics['f1']:.4f}")

        except FileNotFoundError as e:
            print(f"Error: Model file not found - {e}")
        except KeyError as e:
            print(f"Error: Missing key in dataset or model config - {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the TransUNet model.".title())
    parser.add_argument(
        "--dataset",
        type=str,
        default=config_files()["tester"]["dataset"],
        help="Dataset to test on ('train' or 'test')".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config_files()["tester"]["device"],
        help="Device for testing (cuda or cpu)".capitalize(),
    )
    args = parser.parse_args()

    tester = Tester(dataset=args.dataset, device=args.device)
    tester.display_images()
