import os
import sys
import yaml
import torch
import joblib

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
