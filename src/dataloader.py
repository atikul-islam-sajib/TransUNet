import os
import sys
import zipfile
import torch
import argparse

sys.path.append("./src/")


class Loader:
    def __init__(
        self,
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

    def unzip_folder(self):
        pass

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
    pass
