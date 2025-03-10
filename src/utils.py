import os
import sys
import math
import yaml
import torch
import joblib
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
from torchview import draw_graph

matplotlib.use("Agg")

sys.path.append("./src/")


def config_files():
    with open("./config.yml", "r") as config_file:
        return yaml.safe_load(config_file)


def dump_files(value=None, filename=None):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)
    else:
        raise ValueError(
            "Determine the filename and value of a config file".capitalize()
        )


def load_file(filename: str = None):
    if filename is not None:
        return joblib.load(filename)
    else:
        raise ValueError(
            "Please provide a filename to load config data from.".capitalize()
        )


def device_init(device: str = "cuda"):
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        return torch.device(device)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm") != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)


def path_names():
    raw_data_path: str = config_files()["artifacts"]["raw_data_path"]
    processed_data_path: str = config_files()["artifacts"]["processed_data_path"]
    train_models_path: str = config_files()["artifacts"]["train_models"]
    best_model_path: str = config_files()["artifacts"]["best_model"]
    files_path: str = config_files()["artifacts"]["files_path"]
    metrics_path: str = config_files()["artifacts"]["metrics_path"]
    train_images: str = config_files()["artifacts"]["train_images"]
    test_image: str = config_files()["artifacts"]["test_image"]
    image_path: str = config_files()["dataloader"]["image_path"]

    return {
        "raw_data_path": raw_data_path,
        "processed_data_path": processed_data_path,
        "train_models_path": train_models_path,
        "best_model_path": best_model_path,
        "files_path": files_path,
        "metrics_path": metrics_path,
        "train_images": train_images,
        "test_image": test_image,
        "image_path": image_path,
    }


def plot_images(
    original_images: torch.Tensor = None,
    original_masks: torch.Tensor = None,
    predicted_images: torch.Tensor = None,
    predicted: bool = False,
    epoch: int = 0,
    testing: bool = False,
):
    if not predicted:
        processed_data_path = path_names()["processed_data_path"]
        processed_data_path = load_file(
            filename=os.path.join(processed_data_path, "test_dataloader.pkl")
        )
        images, masks = next(iter(processed_data_path))
    else:
        images = original_images
        masks = original_masks

    max_number = min(16, images.size(0))
    num_of_rows = math.ceil(math.sqrt(max_number))
    num_of_columns = math.ceil(max_number / num_of_rows)

    paths = path_names()

    plt.figure(figsize=(10, 20))

    for index, (image, mask) in enumerate(zip(images[:max_number], masks[:max_number])):
        image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        mask = mask.squeeze().detach().cpu().numpy()

        image = (image - image.min()) / (image.max() - image.min())
        mask = (mask - mask.min()) / (mask.max() - mask.min())

        if predicted:
            pred_mask = predicted_images[index].permute(1, 2, 0)
            pred_mask = pred_mask.detach().cpu().numpy()
            pred_mask = (pred_mask - pred_mask.min()) / (
                pred_mask.max() - pred_mask.min()
            )

            plt.subplot(3 * num_of_rows, num_of_columns, 3 * index + 1)
            plt.imshow(image)
            plt.axis("off")
            plt.title("Image")

            plt.subplot(3 * num_of_rows, num_of_columns, 3 * index + 2)
            plt.imshow(mask, cmap="gray")
            plt.axis("off")
            plt.title("Mask")

            plt.subplot(3 * num_of_rows, num_of_columns, 3 * index + 3)
            plt.imshow(pred_mask, cmap="gray")
            plt.axis("off")
            plt.title("Pred Mask")

        else:
            plt.subplot(2 * num_of_rows, num_of_columns, 2 * index + 1)
            plt.imshow(image)
            plt.axis("off")
            plt.title("Image")

            plt.subplot(2 * num_of_rows, num_of_columns, 2 * index + 2)
            plt.imshow(mask, cmap="gray")
            plt.axis("off")
            plt.title("Mask")

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.tight_layout()
    if not predicted:
        save_path = os.path.join(paths["files_path"], "images.png")
    elif predicted and not testing:
        save_path = os.path.join(paths["train_images"], f"image{epoch}.png")
    else:
        save_path = os.path.join(paths["test_image"], f"pred_image{epoch}.png")

    plt.savefig(save_path)
    plt.close()
    print("Image files saved in", save_path)


def total_params(model=None):
    if model is None:
        raise ValueError(
            "Please provide a model to calculate the total parameters.".capitalize()
        )
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_model_architecture(
    model=None,
    input_data: torch.Tensor = None,
    model_name: str = "./artifacts/files_name/",
    format: str = "pdf",
):
    if model is None and not isinstance(input_data, torch):
        raise ValueError(
            "Please provide a model and input data to plot the model architecture.".capitalize()
        )

    filename = path_names()["files_path"]
    draw_graph(model=model, input_data=input_data).visual_graph.render(
        filename=os.path.join(filename, model_name), format=format
    )
    print(f"Model architecture saved in {filename}/{model_name}.{format}")


class IoUScore(nn.Module):
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        super(IoUScore, self).__init__()
        self.threshold = threshold
        self.smooth = smooth

    def forward(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        if not isinstance(predicted, torch.Tensor) or not isinstance(
            actual, torch.Tensor
        ):
            raise TypeError("Inputs must be of type torch.Tensor.")

        predicted = (predicted > self.threshold).float()
        actual = (actual > self.threshold).float()

        intersection = (predicted * actual).sum(dim=(1, 2, 3))
        union = (predicted + actual).sum(dim=(1, 2, 3)) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return iou.mean()
