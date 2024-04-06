# DnCNN Helper Tool

This Python script is designed to facilitate the setup and training of the Deep Convolutional Neural Network (DnCNN) for image denoising tasks. It initializes the DnCNN model, prepares data loaders, sets up the optimizer and loss function based on user-defined arguments, and provides a quick start for training and evaluating the model.

## Features

- Initializes DnCNN model with weight initialization.
- Configures optimizer (Adam or SGD) and loss function (MSELoss or SmoothL1Loss).
- Loads training, testing, and full dataloaders from processed data.
- Flexible device allocation for model computation (CPU, CUDA, MPS).

## Usage

### Arguments

The script accepts the following arguments through keyword arguments (`kwargs`):

- `device`: The device for computation (`cpu`, `cuda`, `mps`). Default is `cpu`.
- `adam`: Boolean, use Adam optimizer if True. Default is True.
- `SGD`: Boolean, use SGD optimizer if True. Ignored if `adam` is True. Default is False.
- `lr`: Learning rate for the optimizer. Default is 0.001.
- `beta1`: Beta1 value for Adam optimizer. Default is 0.9.
- `huber_loss`: Boolean, use SmoothL1Loss if True, else use MSELoss. Default is False.

### Running the Script

To use the script, you can import and call the `helper` function from another Python file or interact with it directly in your code. Here's an example of how to call the `helper` function with custom arguments:

```python
from your_script_name import helper

config = {
    "device": "cuda",
    "adam": True,
    "SGD": False,
    "lr": 0.001,
    "beta1": 0.9,
    "huber_loss": True
}

setup = helper(**config)

# Access the initialized model, optimizer, and dataloaders
model = setup["model"]
optimizer = setup["optimizer"]
train_dataloader = setup["train_dataloader"]
```

### Note

Before running the script, ensure that your processed data is correctly placed in `PROCESSED_DATA_PATH` as defined in your `config.py`. The script expects `.pkl` files for train, test, and combined dataloaders.
