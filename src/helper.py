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
from vgg16 import VGG16
from resnet import ResNet


def helper(**kwargs):
    """
    Initializes and configures the DnCNN model, optimizer, loss criterion, and data loaders.

    This function sets up the DnCNN model with specified training parameters, selects the optimizer based on user
    preferences (Adam or SGD), chooses the loss function (Huber Loss or MSE Loss), and loads the training, testing,
    and complete datasets. It also applies weight initialization to the model and moves the model to the specified
    computation device.

    Parameters:
        device (str): The device to train the model on ('cuda', 'mps', or 'cpu').
        adam (bool): If True, use Adam optimizer; otherwise, check for SGD.
        SGD (bool): If True and adam is False, use SGD optimizer.
        lr (float): Learning rate for the optimizer.
        beta1 (float): Beta1 hyperparameter for the Adam optimizer.
        huber_loss (bool): If True, use Huber loss; otherwise, use MSE loss.
        lr_scheduler (bool): If True, use learning rate scheduler.

    Returns:
        dict: A dictionary containing the initialized model, optimizer, loss criterion, data loaders, and scheduler.

    Raises:
        FileNotFoundError: If the processed data directory does not exist or dataloaders could not be found.

    Note:
        - The function relies on the `params` function to load model-specific parameters like beta1 for Adam, step_size,
          and gamma for the StepLR scheduler, and momentum for SGD from a YAML file.
        - It applies a predefined weight initialization scheme to the DnCNN model before training.
        - All components are configured and returned in a dictionary for easy access.

    Example:
        >>> config = {
                "device": "cuda",
                "adam": True,
                "SGD": False,
                "lr": 1e-3,
                "beta1": 0.9,
                "huber_loss": False,
                "lr_scheduler": True
            }
        >>> setup = helper(**config)
        >>> print(setup["model"])
        >>> print(setup["train_dataloader"].dataset)
    """
    device = kwargs["device"]
    adam = kwargs["adam"]
    is_vgg16 = kwargs["is_vgg16"]
    is_resnet = kwargs["is_resnet"]
    SGD = kwargs["SGD"]
    lr = kwargs["lr"]
    beta1 = kwargs["beta1"]
    huber_loss = kwargs["huber_loss"]
    lr_scheduler = kwargs["lr_scheduler"]

    if is_vgg16:
        model = VGG16()
    elif is_resnet:
        model = ResNet()
    else:
        model = DnCNN()

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
