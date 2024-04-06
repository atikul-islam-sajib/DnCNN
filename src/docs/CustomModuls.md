# DnCNN Training and Testing Guide

This guide provides detailed instructions for training and testing the Deep Convolutional Neural Network (DnCNN) for image denoising, utilizing custom modules designed for MPS, CUDA, and CPU devices.

## Modules Overview

The implementation leverages custom modules for flexible data handling (`Loader`), model training (`Trainer`), and evaluation (`Charts`). These modules support device-specific optimizations to ensure efficient execution across different hardware platforms.

## Arguments

To provide a clear overview of the arguments used in the example setup, below is a table along with explanations for each argument. This table format and descriptions aim to aid in understanding how these arguments influence the training process and model configuration.

| Argument          | Type  | Default Value | Description                                                                                                                                          |
| ----------------- | ----- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `epochs`          | int   | 50            | The total number of training epochs to run. This defines how many times the entire dataset is passed through the network.                            |
| `lr`              | float | 0.0001        | Learning rate for the optimizer. It controls the step size at each iteration while moving toward a minimum of the loss function.                     |
| `device`          | str   | "cuda"        | Specifies the device on which the model will run. Options include "cuda" for NVIDIA GPUs, "mps" for Apple Silicon GPUs, and "cpu" for the CPU.       |
| `display`         | bool  | True          | If set to True, training progress and metrics will be displayed. Otherwise, the training progress will not be printed to the console.                |
| `beta1`           | float | 0.9           | The exponential decay rate for the first moment estimates in the Adam optimizer. It controls the momentum term.                                      |
| `adam`            | bool  | True          | If True, the Adam optimizer is used. If False, and `SGD` is True, the SGD optimizer is used instead.                                                 |
| `SGD`             | bool  | False         | If True (and `adam` is False), the SGD optimizer is used for training.                                                                               |
| `is_lr_scheduler` | bool  | False         | Determines whether a learning rate scheduler is used. A scheduler adjusts the learning rate based on the number of epochs.                           |
| `is_l1`           | bool  | False         | If True, L1 regularization is applied to the model parameters. This encourages sparsity in the model weights.                                        |
| `is_l2`           | bool  | False         | If True, L2 regularization is applied, which helps prevent the model weights from growing too large.                                                 |
| `is_huber_loss`   | bool  | False         | If True, the Huber loss function is used instead of the default loss function. Huber loss is less sensitive to outliers than the squared error loss. |
| `is_weight_clip`  | bool  | False         | Enables weight clipping if set to True. This limits the weights to stay within a specified range to prevent exploding gradients.                     |

## Usage

### Preparing Data

The `Loader` module is responsible for preparing the data for training and testing. It unzips the dataset, resizes images, and splits the data according to the specified ratio.

```python
loader = Loader(
    image_path="./data/data.zip",
    batch_size=32,
    image_size=64,
    split_ratio=0.2,
)
loader.unzip_folder()
dataloader = loader.create_dataloader()
```

### Training the Model

The `Trainer` module handles the model's training process, including setting up the optimizer, loss function, and executing the training loops.

```python
trainer = Trainer(
    epochs=50,
    lr=0.0001,
    device="cuda",  # Change to "cpu" or "mps" as needed
    display=True,
    beta1=0.9,
    adam=True,
    SGD=False,
    is_lr_scheduler=False,
    is_l1=False,
    is_l2=False,
    is_huber_loss=False,
    is_weight_clip=False,
)
trainer.train()
Trainer.display_metrics()
```

### Evaluating the Model

The `Charts` module provides functionality for visualizing the denoising results of the trained model on test images.

```python
charts = Charts(
    device="cuda",  # Change to "cpu" or "mps" as needed
)
charts.plot()
```
