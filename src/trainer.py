import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

sys.path.append("src/")

from config import (
    TRAIN_MODELS_PATH,
    BEST_MODELS_PATH,
    TRAIN_IMAGES_PATH,
    METRICS_PATH,
    FILE_PATH,
)
from utils import device_init, params, dump, load
from helper import helper
from DnCNN import DnCNN


class Trainer:
    """
    Trainer class for the DnCNN model, encapsulating training procedures, loss computation,
    and checkpoint handling.

    Attributes:
        epochs (int): Total number of training epochs.
        lr (float): Learning rate for optimizer.
        beta1 (float): Beta1 parameter for Adam optimizer.
        device (str): Computation device ('cuda', 'mps', 'cpu').
        is_adam (bool): Flag to use Adam optimizer; default is True.
        is_SGD (bool): Flag to use SGD optimizer; ignored if is_adam is True.
        is_l1 (bool): Flag to add L1 regularization.
        is_l2 (bool): Flag to add L2 regularization.
        is_huber_loss (bool): Flag to use Huber loss instead of default.
        is_weight_clip (bool): Flag to clip weights; requires specified min and max in params.
        display (bool): Flag to display training progress and metrics.
        DnCNN (DnCNN): Instance of DnCNN model.
        infinity (float): Variable to track the best loss value.
    """

    def __init__(
        self,
        epochs=100,
        lr=1e-3,
        beta1=0.5,
        device="mps",
        adam=True,
        SGD=False,
        is_lr_scheduler=False,
        is_l1=False,
        is_l2=False,
        is_huber_loss=False,
        is_weight_clip=False,
        display=False,
    ):
        """
        Initializes the trainer with model and training parameters.

        Parameters:
            epochs (int): Number of epochs for training.
            lr (float): Learning rate for the optimizer.
            beta1 (float): Beta1 hyperparameter for Adam optimizer.
            device (str): Device ('cuda', 'mps', 'cpu') for training.
            adam (bool): Flag to use Adam optimizer.
            SGD (bool): Flag to use SGD optimizer; ignored if adam is True.
            is_l1 (bool): Enable L1 regularization.
            is_l2 (bool): Enable L2 regularization.
            is_huber_loss (bool): Use Huber loss as the criterion.
            is_weight_clip (bool): Enable weight clipping.
            display (bool): Enable progress display.
        """
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.is_adam = adam
        self.is_SGD = SGD
        self.device = device
        self.is_lr_scheduler = is_lr_scheduler
        self.is_l1 = is_l1
        self.is_l2 = is_l2
        self.is_huber_loss = is_huber_loss
        self.is_weight_clip = is_weight_clip
        self.display = display
        self.DnCNN = DnCNN()
        self.infinity = float("inf")

        try:
            parameter = helper(
                model=self.DnCNN,
                lr=self.lr,
                adam=self.is_adam,
                lr_scheduler=self.is_lr_scheduler,
                SGD=self.is_SGD,
                device=self.device,
                beta1=self.beta1,
                huber_loss=self.is_huber_loss,
            )

        except Exception as e:
            print(e)

        else:
            self.model = parameter["model"]
            self.optimizer = parameter["optimizer"]
            self.criterion = parameter["criterion"]
            self.train_dataloader = parameter["train_dataloader"]
            self.test_dataloader = parameter["test_dataloader"]
            self.dataloader = parameter["dataloader"]
            self.scheduler = parameter["scheduler"]

        finally:
            self.device = device_init(device=self.device)
            self.history = {"train_loss": [], "test_loss": []}

    def l1(self, **kwargs):
        """
        Applies L1 regularization to the model parameters.

        Parameters:
            kwargs (dict): Contains 'lambda_value' for regularization strength and 'model' for the DnCNN instance.

        Returns:
            torch.Tensor: The L1 regularization term.
        """
        return kwargs["lambda_value"] * sum(
            torch.norm(params, 1) for params in kwargs["model"].parameters()
        )

    def l2(self, **kwargs):
        """
        Applies L2 regularization to the model parameters.

        Parameters:
            kwargs (dict): Contains 'lambda_value' for regularization strength and 'model' for the DnCNN instance.

        Returns:
            torch.Tensor: The L2 regularization term.
        """
        return kwargs["lambda_value"] * sum(
            torch.norm(params, 2) for params in kwargs["model"].parameters()
        )

    def update_train_loss(self, **kwargs):
        """
        Computes and updates the training loss for a single batch.

        Parameters:
            kwargs (dict): Contains 'noise_images' and 'clean_images' tensors.

        Returns:
            float: The training loss value.
        """
        self.optimizer.zero_grad()

        train_loss = self.criterion(
            self.model(kwargs["noise_images"]), kwargs["clean_images"]
        )

        if self.is_l1:
            train_loss + self.l1(
                lambda_value=params()["model"]["lambda"], model=self.model
            )

        if self.is_l2:
            train_loss + self.l2(
                lambda_value=params()["model"]["lambda"], model=self.model
            )

        if self.is_weight_clip:
            for params in self.model.parameters():
                params.data.clamp_(params()["model"]["min"], params()["model"]["max"])

        train_loss.backward()
        self.optimizer.step()

        return train_loss.item()

    def update_test_loss(self, **kwargs):
        """
        Computes the test loss for a single batch.

        Parameters:
            kwargs (dict): Contains 'noise_images' and 'clean_images' tensors.

        Returns:
            float: The test loss value.
        """
        predicted_loss = self.criterion(
            self.model(kwargs["noise_images"]), kwargs["clean_images"]
        )

        return predicted_loss.item()

    def save_checkpoints(self, **kwargs):
        """
        Saves model checkpoints and the best model based on test loss.

        Parameters:
            kwargs (dict): Contains 'epoch' and 'test_loss' for checkpoint information.
        """
        if os.path.exists(TRAIN_MODELS_PATH):
            torch.save(
                self.model.state_dict(),
                os.path.join(TRAIN_MODELS_PATH, "model_{}.pth".format(kwargs["epoch"])),
            )
        else:
            raise Exception("Model cannot be saved".capitalize())

        if kwargs["test_loss"] < self.infinity:
            self.infinity = kwargs["test_loss"]
            if os.path.exists(BEST_MODELS_PATH):
                torch.save(
                    {
                        "epochs": kwargs["epoch"],
                        "loss": kwargs["test_loss"],
                        "model": self.model.state_dict(),
                    },
                    os.path.join(
                        BEST_MODELS_PATH, "best_model_{}.pth".format(kwargs["epoch"])
                    ),
                )

    def display_progress(self, **kwargs):
        """
        Displays or prints the training progress at the end of each epoch.

        Parameters:
            kwargs (dict): Contains 'epoch', 'train_loss', and 'test_loss'.
        """
        if self.display:
            print(
                "Epochs - [{}/{}] - train_loss: {:.4f} - test_loss: {:.4f}".format(
                    kwargs["epoch"],
                    self.epochs,
                    np.array(kwargs["train_loss"]).mean(),
                    np.array(kwargs["test_loss"]).mean(),
                )
            )
        else:
            print("Epochs - [{}/{}] is completed".format(kwargs["epoch"], self.epochs))

    def train(self):
        """
        Executes the training loop over the set number of epochs.

        This method iterates through both training and testing dataloaders for each epoch,
        computes the loss for each batch, and updates the model parameters accordingly. It keeps track of the training
        and testing losses to monitor the model's performance over time. Additionally, it saves model checkpoints and
        the best model based on the test loss. It also visualizes the training progress and optionally saves generated
        images to inspect the model's output quality.

        The method handles exceptions during the checkpoint saving process and ensures that progress is displayed
        and metrics are recorded regardless of intermediate errors. It concludes by saving the training history
        to a specified path for further analysis.

        Raises:
            Exception: If the METRICS_PATH does not exist, indicating an issue with the specified directory for saving metrics.
        """
        for epoch in tqdm(range(self.epochs)):
            train_loss = list()
            test_loss = list()

            self.model.train()

            for _, (noise_images, clean_images) in enumerate(self.train_dataloader):
                noise_images = noise_images.to(self.device)
                clean_images = clean_images.to(self.device)

                train_loss.append(
                    self.update_train_loss(
                        noise_images=noise_images, clean_images=clean_images
                    )
                )

            self.model.eval()

            for _, (noise_images, clean_images) in enumerate(self.test_dataloader):
                noise_images = noise_images.to(self.device)
                clean_images = clean_images.to(self.device)

                test_loss.append(
                    self.update_test_loss(
                        noise_images=noise_images, clean_images=clean_images
                    )
                )

            if self.is_lr_scheduler:
                self.scheduler.step()

            try:
                self.save_checkpoints(
                    epoch=epoch + 1, test_loss=np.array(test_loss).mean()
                )
            except Exception as e:
                print(
                    "The exception caught in the section of saving checkpoints".capitalize()
                )
            else:
                noise_images, clean_images = next(iter(self.dataloader))

                noise_images = noise_images.to(self.device)
                clean_images = clean_images.to(self.device)

                predicted_images = self.model(noise_images)

                save_image(
                    predicted_images,
                    os.path.join(TRAIN_IMAGES_PATH, "image{}.png".format(epoch + 1)),
                    nrow=5,
                    normalize=True,
                )

            finally:
                self.display_progress(
                    epoch=epoch + 1, train_loss=train_loss, test_loss=test_loss
                )

            try:
                self.history["train_loss"].append(np.array(train_loss).mean())
                self.history["test_loss"].append(np.array(test_loss).mean())

            except Exception as e:
                print(e)

        if os.path.exists(METRICS_PATH):
            dump(value=self.history, filename=os.path.join(METRICS_PATH, "metrics.pkl"))
        else:
            raise Exception("Metrics path is not found".capitalize())

    @staticmethod
    def display_metrics():
        """
        Plots the training and testing loss curves from the training history.
        """
        if os.path.exists(METRICS_PATH):
            history = load(filename=os.path.join(METRICS_PATH, "metrics.pkl"))

            plt.plot(history["train_loss"], label="train_loss")
            plt.plot(history["test_loss"], label="test_loss")
            plt.legend()

            try:
                plt.savefig(os.path.join(METRICS_PATH, "metrics.png"))
            except Exception as e:
                print(e)

            else:
                plt.show()
        else:
            raise Exception("Metrics path is not found".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Define the training of DnCNN".title())

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

    args = parser.parse_args()

    if args.train:
        trainer = Trainer(
            epochs=args.epochs,
            lr=args.lr,
            beta1=args.beta1,
            device=args.device,
            display=args.display,
            adam=args.adam,
            SGD=args.SGD,
            is_l1=args.is_l1,
            is_l2=args.is_l2,
            is_huber_loss=args.is_huber_loss,
            is_weight_clip=args.is_weight_clip,
        )

        trainer.train()

        Trainer.display_metrics()

    else:
        raise Exception("The training cannot be done".capitalize())
