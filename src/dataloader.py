import os
import sys
import zipfile
import torch
import warnings
import argparse

sys.path.append("./src/")

from utils import config_files, path_names


class Loader:
    def __init__(
        self,
        image_path=None,
        image_channels: int = 3,
        image_size: int = 224,
        batch_size: int = 4,
        split_size: float = 0.25,
        shuffle: bool = False,
    ):
        self.image_channels = image_channels
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size
        self.shuffle = shuffle

        if image_path is None:
            self.image_path = path_names()["image_path"]
        else:
            warnings.warn(
                "Ensure the provided image path is correct; an incorrect path may result in an error.".title()
            )
            self.image_path = image_path

    def unzip_folder(self):
        if os.path.exists(self.image_path):
            with zipfile.ZipFile(file=self.image_path, mode="r") as zip_file:
                zip_file.extractall(path=path_names()["processed_data_path"])
            print(
                f"Dataset unzipped successfully in {path_names()["processed_data_path"]}".capitalize()
            )
        else:
            raise FileNotFoundError("Image file not found".capitalize())

    def split_dataset(self):
        pass

    def features_extracted(self):
        pass

    def create_dataloader(self):
        pass

    @staticmethod
    def display_images():
        pass

    @staticmethod
    def dataset_details():
        pass


if __name__ == "__main__":
    loader = Loader(
        image_path="./data/raw/dataset.zip",
        image_channels=3,
        image_size=224,
        batch_size=4,
        split_size=0.25,
        shuffle=False,
    )
    loader.unzip_folder()
