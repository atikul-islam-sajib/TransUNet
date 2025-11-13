import sys
import argparse
import torch.nn as nn

sys.path.append("./src/")

from tester import Tester
from trainer import Trainer
from dataloader import Loader
from utils import config_files, path_names


def cli():
    dataloader_cfg = config_files()["dataloader"]
    trainer_cfg = config_files()["trainer"]
    tester_cfg = config_files()["tester"]

    image_path = path_names()["image_path"]
    image_channels = dataloader_cfg["image_channels"]
    image_size = dataloader_cfg["image_size"]
    batch_size = dataloader_cfg["batch_size"]
    split_size = dataloader_cfg["split_size"]

    epochs = trainer_cfg["epochs"]
    lr = trainer_cfg["lr"]
    optimizer = trainer_cfg["optimizer"]
    optim_cfg = trainer_cfg["optimizers"]

    beta1 = beta2 = weight_decay = momentum = 0.0

    if optimizer == "Adam":
        beta1 = float(optim_cfg["Adam"]["beta1"])
        beta2 = float(optim_cfg["Adam"]["beta2"])
        weight_decay = float(optim_cfg["Adam"]["weight_decay"])

    elif optimizer == "AdamW":
        beta1 = float(optim_cfg["AdamW"]["beta1"])
        beta2 = float(optim_cfg["AdamW"]["beta2"])
        weight_decay = float(optim_cfg["AdamW"]["weight_decay"])

    elif optimizer == "SGD":
        momentum = float(optim_cfg["SGD"]["momentum"])
        weight_decay = float(optim_cfg["SGD"]["weight_decay"])

    loss_cfg = trainer_cfg["loss"]
    loss_type = loss_cfg["type"]
    loss_smooth = float(loss_cfg["loss_smooth"])
    alpha_focal = float(loss_cfg["alpha_focal"])
    gamma_focal = float(loss_cfg["gamma_focal"])
    alpha_tversky = float(loss_cfg["alpha_tversky"])
    beta_tversky = float(loss_cfg["beta_tversky"])

    l1_regularization = trainer_cfg["l1_regularization"]
    elastic_net_regularization = trainer_cfg["elastic_net_regularization"]
    verbose = trainer_cfg["verbose"]
    device = trainer_cfg["device"]

    use_diff_attn = trainer_cfg.get("use_diff_attn", False)
    lam = float(trainer_cfg.get("lam", 0.5))

    parser = argparse.ArgumentParser(
        description="CLI configuration for training/testing TransUNet".title()
    )

    parser.add_argument("--image_path", type=str, default=image_path)
    parser.add_argument("--image_channels", type=int, default=image_channels)
    parser.add_argument("--image_size", type=int, default=image_size)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--split_size", type=float, default=split_size)

    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--beta1", type=float, default=beta1)
    parser.add_argument("--beta2", type=float, default=beta2)
    parser.add_argument("--weight_decay", type=float, default=weight_decay)
    parser.add_argument("--momentum", type=float, default=momentum)

    parser.add_argument("--loss_smooth", type=float, default=loss_smooth)
    parser.add_argument("--alpha_focal", type=float, default=alpha_focal)
    parser.add_argument("--gamma_focal", type=float, default=gamma_focal)
    parser.add_argument("--alpha_tversky", type=float, default=alpha_tversky)
    parser.add_argument("--beta_tversky", type=float, default=beta_tversky)
    parser.add_argument("--loss_type", type=str, default=loss_type)

    parser.add_argument(
        "--optimizer", type=str, default=optimizer, choices=["Adam", "AdamW", "SGD"]
    )

    parser.add_argument("--l1_regularization", action="store_true")
    parser.add_argument("--elastic_net_regularization", action="store_true")

    parser.add_argument("--verbose", action="store_true", default=verbose)

    parser.add_argument(
        "--use_diff_attn",
        action="store_true",
        help="Enable differential attention in ViT encoder",
    )
    parser.add_argument(
        "--lam", type=float, default=lam, help="Lambda for differential attention"
    )

    parser.add_argument("--dataset_type", type=str, default=tester_cfg["dataset"])
    parser.add_argument("--device", type=str, default=device)

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)

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
            loss_func=args.loss_type,
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
            use_diff_attn=args.use_diff_attn,   
            lam=args.lam,                      
        )

        trainer.train()
        Trainer.display_history()

    elif args.test:
        tester = Tester(
            dataset=args.dataset_type,
            device=args.device,
        )
        tester.display_images()


if __name__ == "__main__":
    cli()
