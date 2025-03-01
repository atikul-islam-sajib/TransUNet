import os
import sys
import cv2
import zipfile
import torch
import warnings
import argparse
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

sys.path.append("./src/")

from utils import config_files, path_names, dump_files


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

        self.train_images = list()
        self.train_masks = list()
        self.test_images = list()
        self.test_masks = list()

    def transform_images(self, type: str = "image"):
        if type == "image":
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.CenterCrop((self.image_size, self.image_size)),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        elif type == "mask":
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.CenterCrop((self.image_size, self.image_size)),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

    def unzip_folder(self):
        if os.path.exists(self.image_path):
            with zipfile.ZipFile(file=self.image_path, mode="r") as zip_file:
                zip_file.extractall(path=path_names()["processed_data_path"])
            print(
                f"Dataset unzipped successfully in {path_names()['processed_data_path']}"
            )

        else:
            raise FileNotFoundError("Image file not found".capitalize())

    def features_extracted(self):
        processed_path = path_names()["processed_data_path"]
        train_path = os.path.join(processed_path, "dataset", "train")
        valid_path = os.path.join(processed_path, "dataset", "test")

        train_images_path = os.path.join(train_path, "image")
        train_masks_path = os.path.join(train_path, "mask")

        test_images_path = os.path.join(valid_path, "image")
        test_masks_path = os.path.join(valid_path, "mask")

        for type, path in tqdm(
            [("train", train_images_path), ("test", test_images_path)],
            desc="Extracted features ..",
        ):
            try:
                mask_path = train_masks_path if type == "train" else test_masks_path

                for image in os.listdir(path):
                    try:
                        if image in os.listdir(mask_path):
                            image_path = os.path.join(path, image)
                            mask_image_path = os.path.join(mask_path, image)

                            _image = cv2.imread(image_path)
                            if _image is None:
                                raise FileNotFoundError(
                                    f"Image file not found: {image_path}"
                                )
                            _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
                            _image = Image.fromarray(_image)
                            _image = self.transform_images(type="image")(_image)

                            _mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
                            _mask = Image.fromarray(_mask)
                            _mask = self.transform_images(type="mask")(_mask)

                            if _mask is None:
                                raise FileNotFoundError(
                                    f"Mask file not found: {mask_image_path}"
                                )

                            if type == "train":
                                self.train_images.append(_image)
                                self.train_masks.append(_mask)

                            elif type == "test":
                                self.test_images.append(_image)
                                self.test_masks.append(_mask)

                    except FileNotFoundError as e:
                        print(f"[WARNING] {e}")
                    except cv2.error as e:
                        print(f"[ERROR] OpenCV error while processing {image}: {e}")
                    except Exception as e:
                        print(f"[ERROR] Unexpected error while processing {image}: {e}")

            except Exception as e:
                print(f"[CRITICAL] Failed to process dataset for {type} images: {e}")

        assert len(self.train_images) == len(
            self.train_masks
        ), "Images, Masks should be equal in size".capitalize()

        assert len(self.test_images) == len(
            self.test_masks
        ), "Images, Masks should be equal in size".capitalize()

        return {
            "train_images": self.train_images,
            "train_masks": self.train_masks,
            "test_images": self.test_images,
            "test_masks": self.test_masks,
        }

    def create_dataloader(self):
        try:
            dataset = self.features_extracted()

            train_dataloader = DataLoader(
                dataset=list(zip(dataset["train_images"], dataset["train_masks"])),
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            )

            valid_dataloader = DataLoader(
                dataset=list(zip(dataset["test_images"], dataset["test_masks"])),
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            )

            for filename, value in [
                ("train_dataloader.pkl", train_dataloader),
                ("test_dataloader.pkl", valid_dataloader),
            ]:
                dump_files(
                    value=value,
                    filename=os.path.join(
                        path_names()["processed_data_path"], filename
                    ),
                )
                print(f"Dataloader saved as {filename}".capitalize())
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")
        except cv2.error as e:
            print(f"[ERROR] OpenCV error while processing: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error while processing: {e}")

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
    loader.create_dataloader()
