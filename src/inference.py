import os
import sys
import cv2
import torch
import argparse
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

sys.path.append("./src/")

from utils import path_names, config_files, device_init
from transUNet import TransUNet


class Inference:
    def __init__(self, image: str = None):
        try:
            self.image = image
            self.paths = config_files()
            self.dataloader = self.paths["dataloader"]
            self.image_size = self.dataloader["image_size"]
            self.device = device_init(device=self.paths["trainer"]["device"])
        except KeyError as e:
            raise ValueError(f"Missing configuration key: {e}")
        except Exception as e:
            raise RuntimeError(f"Error during initialization: {e}")

    def image_preprocess(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.CenterCrop((self.image_size, self.image_size)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def load_model(self):
        try:
            config = config_files()
            model = TransUNet(
                image_channels=config["dataloader"]["image_channels"],
                image_size=config["dataloader"]["image_size"],
                nheads=config["TransUNet"]["nheads"],
                num_layers=config["TransUNet"]["num_layers"],
                dim_feedforward=config["TransUNet"]["dim_feedforward"],
                dropout=float(config["TransUNet"]["dropout"]),
                activation=config["TransUNet"]["activation"],
                layer_norm_eps=float(config["TransUNet"]["layer_norm_eps"]),
                bias=config["TransUNet"]["bias"],
            )

            model.to(self.device)

            best_model_path = os.path.join(
                path_names()["best_model_path"], "best_model.pth"
            )
            if not os.path.exists(best_model_path):
                raise FileNotFoundError(
                    f"Model checkpoint not found: {best_model_path}"
                )

            state_dict = torch.load(best_model_path, map_location=self.device)

            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            else:
                model.load_state_dict(state_dict)

            model.eval()
            return model

        except FileNotFoundError as e:
            raise RuntimeError(f"Model loading error: {e}")
        except KeyError as e:
            raise ValueError(f"Invalid model checkpoint format. Missing key: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during model loading: {e}")

    def predict(self, saved_image_path: str = None):
        """Run inference on the input image and save the predicted output."""
        try:
            if not os.path.exists(self.image):
                raise FileNotFoundError(f"Input image not found: {self.image}")

            image = cv2.imread(self.image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = self.image_preprocess()(image)
            image = image.unsqueeze(0).to(self.device)

            model = self.load_model()

            with torch.no_grad():
                predicted = model(image)

            if saved_image_path is None:
                inference_dir = os.path.join("./artifacts/outputs", "Inference")
            else:
                inference_dir = saved_image_path

            os.makedirs(inference_dir, exist_ok=True)

            saved_image_path = os.path.join(inference_dir, "inference.png")
            save_image(predicted, saved_image_path)
            print(f"Prediction saved at: {saved_image_path}")

        except FileNotFoundError as e:
            print(f"File error: {e}")
        except RuntimeError as e:
            print(f"Runtime error during inference: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference on a given image.".title()
    )
    parser.add_argument(
        "--image",
        type=str,
        default=config_files()["Inference"]["Image"],
        help="Path to the input image".capitalize(),
    )
    args = parser.parse_args()
    try:
        inference = Inference(image=args.image)
        inference.predict()
    except Exception as e:
        print(f"Fatal error: {e}")
