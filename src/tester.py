import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from transUNet import TransUNet
from utils import (
    config_files,
    device_init,
    plot_images,
    load_file,
    path_names,
)

class Tester:
    def __init__(self, dataset: str = "test", device: str = "cuda"):
        self.dataset = dataset
        self.device = device

        self.device = device_init(device=self.device)

    def load_dataloder(self):
        processed_path = path_names()["processed_data_path"]
        train_dataloader_path = os.path.join(processed_path, "train_dataloader.pkl")
        valid_dataloader_path = os.path.join(processed_path, "test_dataloader.pkl")

        train_dataloader = load_file(filename=train_dataloader_path)
        valid_dataloader = load_file(filename=valid_dataloader_path)

        return {
            "train_dataloader": train_dataloader,
            "valid_dataloader": valid_dataloader,
        }

    def display_images(self):
        try:
            dataset = self.load_dataloder()
            if self.dataset not in ["train", "test"]:
                raise ValueError(
                    "Invalid dataset. Please choose between 'train' or 'valid'."
                )

            dataloader = (
                dataset["train_dataloader"]
                if self.dataset == "train"
                else dataset["valid_dataloader"]
            )

            best_model_path = os.path.join(
                path_names()["best_model_path"], "best_model.pth"
            )
            model_state = torch.load(best_model_path, weights_only=False)[
                "model_state_dict"
            ]

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
            )

            model.to(self.device)
            model.load_state_dict(model_state)

            for index, (images, masks) in enumerate(dataloader):
                original_images = images.to(self.device)
                original_masks = masks.to(self.device)
                predicted_images = model(original_images)

                plot_images(
                    original_images=original_images,
                    original_masks=original_masks,
                    predicted_images=predicted_images,
                    predicted=True,
                    epoch=index + 1,
                    testing=True,
                )

        except FileNotFoundError as e:
            print(f"Error: Model file not found - {e}")
        except KeyError as e:
            print(f"Error: Missing key in dataset or model config - {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model.".title())
    parser.add_argument(
        "--dataset",
        type=str,
        default=config_files()["tester"]["dataset"],
        help="Dataset to test on".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config_files()["tester"]["device"],
        help="Device to use for testing".capitalize(),
    )
    args = parser.parse_args()

    dataset = args.dataset
    device = args.device

    tester = Tester(dataset=dataset, device=device)

    tester.display_images()
