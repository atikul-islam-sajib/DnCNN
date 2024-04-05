import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("src/")

from utils import weight_init, load
from config import PROCESSED_DATA_PATH


def helper(*kwargs):
    model = kwargs["model"]
    device = kwargs["device"]
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]
    lr = kwargs["lr"]
    beta1 = kwargs["beta1"]
    huber_loss = kwargs["huber_loss"]

    if adam:
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    elif SGD:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    if huber_loss:
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()

    if os.path.exists(PROCESSED_DATA_PATH):
        train_dataloader, test_dataloader = load(
            os.path.join(PROCESSED_DATA_PATH, "train_dataloader.pkl")
        ), load(os.path.join(PROCESSED_DATA_PATH, "test_dataloader.pkl"))
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
    }
