import sys
import argparse
import torch.nn as nn

sys.path.append("./src/")

from tester import Tester
from trainer import Trainer
from dataloader import Loader
from utils import config_files, path_names


def cli():
    image_path = path_names()["image_path"]
    image_channels = config_files()["dataloader"]["image_channels"]
    image_size = config_files()["dataloader"]["image_size"]
    batch_size = config_files()["dataloader"]["batch_size"]
    split_size = config_files()["dataloader"]["split_size"]
    trainer_config = config_files()["trainer"]

    epochs = trainer_config["epochs"]
    lr = trainer_config["lr"]

    optimizer = trainer_config["optimizer"]
    optmizers_config = trainer_config["optimizers"]

    beta1, beta2, weight_decay, momentum = 0.0, 0.0, 0.0, 0.0

    if "Adam" in optmizers_config or "AdamW" in optmizers_config:
        beta1 = float(optmizers_config["Adam"]["beta1"])
        beta2 = float(optmizers_config["Adam"]["beta2"])
        weight_decay = float(optmizers_config["Adam"]["weight_decay"])

    elif "SGD" in optmizers_config:
        momentum = float(optmizers_config["SGD"]["momentum"])
        weight_decay = float(optmizers_config["SGD"]["weight_decay"])

    loss = trainer_config["loss"]
    loss_type = loss["type"]
    loss_smooth = float(loss["loss_smooth"])
    alpha_focal = float(loss["alpha_focal"])
    gamma_focal = float(loss["gamma_focal"])
    alpha_tversky = float(loss["alpha_tversky"])
    beta_tversky = float(loss["beta_tversky"])
    l1_regularization = trainer_config["l1_regularization"]
    elastic_net_regularization = trainer_config["elastic_net_regularization"]
    verbose = trainer_config["verbose"]
    device = trainer_config["device"]

    parser = argparse.ArgumentParser(
        description="CLI configuration for the TransUNet 4 train and test".title()
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=image_path,
        help="Path to the dataset image file (zip format).".capitalize(),
    )
    parser.add_argument(
        "--image_channels",
        type=int,
        default=image_channels,
        help="Number of channels in the dataset image.".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=image_size,
        help="Size of the dataset image.".capitalize(),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=batch_size,
        help="Batch size for training and validation.".capitalize(),
    )
    parser.add_argument(
        "--split_size",
        type=float,
        default=split_size,
        help="Percentage of data to be used for validation.".capitalize(),
    )

    parser.add_argument(
        "--epochs", type=int, default=epochs, help="Number of epochs for training"
    )
    parser.add_argument(
        "--lr", type=float, default=lr, help="Learning rate for training"
    )
    parser.add_argument(
        "--beta1", type=float, default=beta1, help="Beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=beta2, help="Beta2 for Adam optimizer"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=weight_decay,
        help="Weight decay for Adam optimizer",
    )
    parser.add_argument(
        "--momentum", type=float, default=momentum, help="Momentum for SGD optimizer"
    )
    parser.add_argument(
        "--loss_smooth", type=str, default=loss_smooth, help="Loss smooth type"
    )
    parser.add_argument(
        "--alpha_focal", type=float, default=alpha_focal, help="Alpha for Focal loss"
    )
    parser.add_argument(
        "--gamma_focal", type=float, default=gamma_focal, help="Gamma for Focal loss"
    )
    parser.add_argument(
        "--alpha_tversky",
        type=float,
        default=alpha_tversky,
        help="Alpha for Tversky loss",
    )
    parser.add_argument(
        "--beta_tversky", type=float, default=beta_tversky, help="Beta for Tversky loss"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=optimizer,
        help="Select the optimizer to use",
        choices=["Adam", "AdamW", "SGD"],
    )
    parser.add_argument(
        "--l1_regularization",
        type=bool,
        default=l1_regularization,
        help="Use L1 regularization",
    )
    parser.add_argument(
        "--elastic_net_regularization",
        type=bool,
        default=elastic_net_regularization,
        help="Use Elastic net regularization",
    )
    parser.add_argument(
        "--verbose", type=bool, default=verbose, help="Display progress and save images"
    )
    parser.add_argument(
        "--loss_type", type=str, default=loss_type, help="Define the loss type"
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default=config_files()["tester"]["dataset"],
        help="Dataset to test on".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config_files()["trainer"]["device"],
        help="Device to use for testing".capitalize(),
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="To Train the model".capitalize(),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="To Test the model".capitalize(),
    )

    args = parser.parse_args()

    if args.train:
        loader = Loader(
            image_path=args.image_path,
            image_channels=args.image_channels,
            image_size=args.image_size,
            batch_size=args.batch_size,
            split_size=args.split_size,
            shuffle=True,
        )

        loader.unzip_folder()
        loader.create_dataloader()

        Loader.display_images()
        Loader.dataset_details()

        trainer = Trainer(
            model=None,
            epochs=args.epochs,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            loss_func=loss_type,
            loss_smooth=args.loss_smooth,
            alpha_focal=args.alpha_focal,
            gamma_focal=args.gamma_focal,
            alpha_tversky=args.alpha_tversky,
            beta_tversky=args.beta_tversky,
            optimizer=args.optimizer,
            l1_regularization=args.l1_regularization,
            elastic_net_regularization=args.elastic_net_regularization,
            verbose=args.verbose,
            device=args.device,
        )

        trainer.train()
        Trainer.display_history()

    elif args.test:
        dataset = args.dataset_type
        device = args.device

        tester = Tester(dataset=dataset, device=device)

        tester.display_images()


if __name__ == "__main__":
    cli()
