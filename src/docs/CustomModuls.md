# DnCNN Training and Testing Guide

This guide provides detailed instructions for training and testing the Deep Convolutional Neural Network (DnCNN) for image denoising, utilizing custom modules designed for MPS, CUDA, and CPU devices.

## Modules Overview

The implementation leverages custom modules for flexible data handling (`Loader`), model training (`Trainer`), and evaluation (`Charts`). These modules support device-specific optimizations to ensure efficient execution across different hardware platforms.

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
