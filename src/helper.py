import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

sys.path.append("src/")

from utils import weight_init, load, params
from config import PROCESSED_DATA_PATH
from DnCNN import DnCNN


def helper(**kwargs):
    model = DnCNN()
    device = kwargs["device"]
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]
    lr = kwargs["lr"]
    beta1 = kwargs["beta1"]
    huber_loss = kwargs["huber_loss"]
    lr_scheduler = kwargs["lr_scheduler"]

    if adam:
        optimizer = optim.Adam(
            model.parameters(), lr=lr, betas=(beta1, params()["model"]["beta1"])
        )
        scheduler = StepLR(
            optimizer=optimizer,
            step_size=params()["model"]["step_size"],
            gamma=params()["model"]["gamma"],
        )
    elif SGD:
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=params()["model"]["momentum"]
        )
        scheduler = StepLR(
            optimizer=optimizer,
            step_size=params()["model"]["step_size"],
            gamma=params()["model"]["gamma"],
        )

    if huber_loss:
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()

    if os.path.exists(PROCESSED_DATA_PATH):
        train_dataloader = load(
            os.path.join(PROCESSED_DATA_PATH, "train_dataloader.pkl")
        )
        test_dataloader = load(os.path.join(PROCESSED_DATA_PATH, "test_dataloader.pkl"))
        dataloader = load(os.path.join(PROCESSED_DATA_PATH, "dataloader.pkl"))

    else:
        raise FileNotFoundError("Could not find processed data")

    try:
        model = model.to(device)
    except Exception as e:
        print("The exception caught in the section is", e)
    else:
        model.apply(weight_init)

    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "dataloader": dataloader,
        "scheduler": scheduler,
    }


if __name__ == "__main__":
    check = helper()

    data, label = next(iter(check["train_dataloader"]))

    print(data.shape)
