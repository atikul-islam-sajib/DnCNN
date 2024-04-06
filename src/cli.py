import sys
import os
import argparse

sys.path.append("src/")

from dataloader import Loader
from DnCNN import DnCNN
from trainer import Trainer
from test import Charts


def cli():
    """
    Command-Line Interface for the Deep Convolutional Neural Network (DnCNN) for image denoising.

    This CLI supports various operations including unzipping the dataset, preparing data loaders,
    training the DnCNN model, and evaluating the model's performance through visualization of
    denoising results on test images.

    Arguments:
    --image_path : str
        Path to the zip file containing the dataset. Defaults to './data/data.zip'.
    --batch_size : int
        Batch size for training and testing. Defaults to 16.
    --image_size : int
        The size to which the images will be resized. Defaults to 64.
    --split_ratio : float
        The ratio of the dataset to be used as the test set. Defaults to 0.2.
    --epochs : int
        Number of training epochs. Defaults to 100.
    --lr : float
        Learning rate for the optimizer. Defaults to 1e-4.
    --device : str
        Computing device ('cuda', 'cpu', 'mps'). Defaults to 'mps'.
    --display : bool
        Whether to display training progress and metrics. Defaults to True.
    --beta1 : float
        Beta1 hyperparameter for Adam optimizer. Defaults to 0.9.
    --train : store_true
        Flag to initiate model training.
    --test : store_true
        Flag to initiate model testing and visualization.
    --adam : bool
        Whether to use Adam optimizer. Defaults to True.
    --SGD : bool
        Whether to use Stochastic Gradient Descent (SGD) optimizer. Defaults to False.
    --is_l1 : bool
        Whether to use L1 regularization. Defaults to False.
    --is_l2 : bool
        Whether to use L2 regularization. Defaults to False.
    --is_huber_loss : bool
        Whether to use Huber loss as the training criterion. Defaults to False.
    --is_weight_clip : bool
        Whether to apply weight clipping. Defaults to False.

    Example usage:
    python script_name.py --train --epochs 50 --batch_size 32 --lr 0.0001
    python script_name.py --test --device cpu
    """
    parser = argparse.ArgumentParser(description="Define the CLI for DnCNN".title())

    parser.add_argument(
        "--image_path",
        type=str,
        default="./data/data.zip",
        help="Path to the zip file containing the dataset".capitalize(),
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size".capitalize()
    )
    parser.add_argument(
        "--image_size", type=int, default=64, help="Image size".capitalize()
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.2,
        help="Spit the dataset".capitalize(),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Define the number of epochs".capitalize(),
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Define the learning rate".capitalize()
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Define the device".capitalize()
    )
    parser.add_argument(
        "--display", type=bool, default=True, help="Define the display".capitalize()
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="Define the beta1".capitalize()
    )
    parser.add_argument(
        "--train", action="store_true", help="Define the beta2".capitalize()
    )
    parser.add_argument(
        "--adam", type=bool, default=True, help="Define the adam".capitalize()
    )
    parser.add_argument(
        "--SGD", type=bool, default=False, help="Define the SGD".capitalize()
    )
    parser.add_argument(
        "--is_l1", type=bool, default=False, help="Define the l1".capitalize()
    )
    parser.add_argument(
        "--is_l2", type=bool, default=False, help="Define the l2".capitalize()
    )
    parser.add_argument(
        "--is_huber_loss",
        type=bool,
        default=False,
        help="Define the huber loss".capitalize(),
    )
    parser.add_argument(
        "--is_weight_clip",
        type=bool,
        default=False,
        help="Define the weight clip".capitalize(),
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the model".capitalize()
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model".capitalize()
    )

    args = parser.parse_args()

    if args.train:

        try:
            loader = Loader(
                image_path=args.image_path,
                batch_size=args.batch_size,
                image_size=args.image_size,
                split_ratio=args.split_ratio,
            )
        except Exception as e:
            print(e)
        else:
            loader.unzip_folder()
            dataloader = loader.create_dataloader()

        try:
            trainer = Trainer(
                epochs=args.epochs,
                lr=args.lr,
                device=args.device,
                display=args.display,
                beta1=args.beta1,
                adam=args.adam,
                SGD=args.SGD,
                is_l1=args.is_l1,
                is_l2=args.is_l2,
                is_huber_loss=args.is_huber_loss,
                is_weight_clip=args.is_weight_clip,
            )
        except Exception as e:
            print(e)
        else:
            trainer.train()

            Trainer.display_metrics()

    elif args.test:
        try:
            charts = Charts(
                device=args.device,
            )
        except Exception as e:
            print(e)
        else:
            charts.plot()


if __name__ == "__main__":
    cli()
