import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("src/")

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
        self.is_adam = adam
        self.is_SGD = SGD
        self.device = device
        self.is_l1 = is_l1
        self.is_l2 = is_l2
        self.is_huber_loss = is_huber_loss
        self.is_weight_clip = is_weight_clip
        self.display = display

        try:
            params = helper(
                lr=self.lr,
                adam=self.is_adam,
                SGD=self.is_SGD,
                device=self.device,
                model=DnCNN(),
                huber_loss=self.is_huber_loss,
            )

        except Exception as e:
            print("The error in the section of helpers is: ", e)

        else:
            self.model = params["model"]
            self.optimizer = params["optimizer"]
            self.criterion = params["criterion"]

        finally:
            self.history = {"train_loss": [], "test_loss": []}

    def l1(self, **kwargs):
        return kwargs["lambda"] * sum(
            torch.norm(params, 1) for params in kwargs["model"].parameters()
        )

    def l2(self, **kwargs):
        return kwargs["lambda"] * sum(
            torch.norm(params, 2) for params in kwargs["model"].parameters()
        )

    def train(self):
        pass
