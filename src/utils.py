import os
import sys
import yaml
import torch
import joblib
import torch.nn as nn

sys.path.append("./src/")


def config_files():
    with open("./config.yml", "r") as config_file:
        return yaml.safe_load(config_file)


def dump_file(value=None, filename=None):
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


print(path_names()["image_path"])
