import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
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
    def __init__(
        self,
        epochs=100,
        lr=1e-3,
        beta1=0.5,
        device="mps",
        adam=True,
        SGD=False,
        is_l1=False,
        is_l2=False,
        is_huber_loss=False,
        is_weight_clip=False,
        display=False,
    ):
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.is_adam = adam
        self.is_SGD = SGD
        self.device = device
        self.is_l1 = is_l1
        self.is_l2 = is_l2
        self.is_huber_loss = is_huber_loss
        self.is_weight_clip = is_weight_clip
        self.display = display
        self.DnCNN = DnCNN()
        self.infinity = float("inf")

        try:
            print("check")
            parameter = helper(
                model=self.DnCNN,
                lr=self.lr,
                adam=self.is_adam,
                SGD=self.is_SGD,
                device=self.device,
                beta1=self.beta1,
                huber_loss=self.is_huber_loss,
            )

        except Exception as e:
            print("except")
            print(e)

        else:
            self.model = parameter["model"]
            self.optimizer = parameter["optimizer"]
            self.criterion = parameter["criterion"]
            self.train_dataloader = parameter["train_dataloader"]
            self.test_dataloader = parameter["test_dataloader"]
            self.dataloader = parameter["dataloader"]

        finally:
            self.device = device_init(device=self.device)
            self.history = {"train_loss": [], "test_loss": []}

    def l1(self, **kwargs):
        return kwargs["lambda_value"] * sum(
            torch.norm(params, 1) for params in kwargs["model"].parameters()
        )

    def l2(self, **kwargs):
        return kwargs["lambda_value"] * sum(
            torch.norm(params, 2) for params in kwargs["model"].parameters()
        )

    def update_train_loss(self, **kwargs):
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
        predicted_loss = self.criterion(
            self.model(kwargs["noise_images"]), kwargs["clean_images"]
        )

        return predicted_loss.item()

    def save_checkpoints(self, **kwargs):
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
        for epoch in tqdm(range(self.epochs)):
            train_loss = list()
            test_loss = list()

            for _, (noise_images, clean_images) in enumerate(self.train_dataloader):
                noise_images = noise_images.to(self.device)
                clean_images = clean_images.to(self.device)

                train_loss.append(
                    self.update_train_loss(
                        noise_images=noise_images, clean_images=clean_images
                    )
                )

            for _, (noise_images, clean_images) in enumerate(self.test_dataloader):
                noise_images = noise_images.to(self.device)
                clean_images = clean_images.to(self.device)

                test_loss.append(
                    self.update_test_loss(
                        noise_images=noise_images, clean_images=clean_images
                    )
                )

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