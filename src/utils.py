import os
import sys
import math
import yaml
import torch
import joblib
import torch.nn as nn
import matplotlib.pyplot as plt

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
    predicted_images: torch.Tensor = None,
    predicted: bool = False,
):
    processed_data_path = path_names()["processed_data_path"]
    processed_data_path = load_file(
        filename=os.path.join(processed_data_path, "test_dataloader.pkl")
    )
    images, masks = next(iter(processed_data_path))

    max_number = min(16, images.size(0))
    num_of_rows = math.ceil(math.sqrt(max_number))
    num_of_columns = math.ceil(max_number / num_of_rows)

    plt.figure(figsize=(10, 20))

    for index, (image, mask) in enumerate(zip(images[:max_number], masks[:max_number])):
        image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        mask = mask.squeeze().detach().cpu().numpy()

        image = (image - image.min()) / (image.max() - image.min())
        mask = (mask - mask.min()) / (mask.max() - mask.min())

        if predicted:
            pred_mask = predicted_images[index].squeeze().permute(1, 2, 0)
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

    save_path = os.path.join(path_names()["files_path"], "images.png")
    plt.savefig(save_path)
    plt.show()
    plt.close()
    print("Image files saved in", save_path)
