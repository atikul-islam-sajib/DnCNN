import sys
import os
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params


class DnCNN(nn.Module):
    """
    Implements the DnCNN model for image denoising.

    The model is composed of a convolutional block followed by several repetitive blocks and an output block. The number of repetitive blocks and their configurations are defined by parameters loaded from a YAML file.

    Attributes:
        image_size (int): Defines the number of filters in the convolutional layers.
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        out_channels (int): Number of output channels. Typically, this will be the same as `in_channels`.
        kernel_size (int): Kernel size for the convolutional layers, loaded from parameters.
        stride (int): Stride for the convolutional layers, loaded from parameters.
        padding (int): Padding for the convolutional layers, loaded from parameters.
        num_repetitive (int): Number of repetitive blocks in the model, loaded from parameters.
        bias (bool): Whether to use bias in the convolutional layers, loaded from parameters.

    Methods:
        ConvBlock: Constructs the initial convolutional block of the model.
        RepetitiveBlock: Constructs the middle repetitive blocks of the model.
        OutputBlock: Constructs the final output block of the model.
        forward: Defines the forward pass of the model.
        total_params: Static method to calculate the total number of parameters in the model.

    Raises:
        ValueError: If there is an issue extracting parameters from the YAML file.

    Example:
        >>> model = DnCNN(image_size=64, in_channels=3, out_channels=3)
        >>> print(DnCNN.total_params(model))
    """

    def __init__(self, in_channels=3, out_channels=3, image_size=64):
        """
        Initializes the DnCNN model with specified image size, input channels, and output channels.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            image_size (int): The image size, which also dictates the number of filters in convolutional layers.
        """
        super(DnCNN, self).__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        try:
            self.kernel_size = params()["model"]["kernel_size"]
            self.stride = params()["model"]["stride"]
            self.padding = params()["model"]["padding"]
            self.num_repetitive = params()["model"]["num_repetitive"]
            self.bias = params()["model"]["bias"]

        except ValueError as e:
            print("Cannot be extract default params fom the yaml file".capitalize())

        else:
            self.layers = list()

            self.conv_block = self.ConvBlock()
            self.repetitive_block = self.RepetitiveBlock()
            self.output_block = self.OutputBlock()

    def ConvBlock(self):
        """
        Constructs the initial convolutional block of the model.

        Returns:
            nn.Sequential: A PyTorch sequential model constituting the convolutional block.
        """
        self.layers = list()

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.image_size,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    bias=self.bias,
                ),
                nn.ReLU(inplace=True),
            )
        )

        return nn.Sequential(*self.layers)

    def RepetitiveBlock(self):
        """
        Constructs the middle repetitive blocks of the model.

        Returns:
            nn.Sequential: A PyTorch sequential model constituting the repetitive blocks.
        """
        self.layers = list()

        for _ in range(self.num_repetitive):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.image_size,
                        out_channels=self.image_size,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        bias=self.bias,
                    ),
                    nn.BatchNorm2d(self.image_size),
                    nn.ReLU(inplace=True),
                )
            )

        return nn.Sequential(*self.layers)

    def OutputBlock(self):
        """
        Constructs the final output block of the model.

        Returns:
            nn.Sequential: A PyTorch sequential model constituting the output block.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=self.image_size,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            ),
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            torch.Tensor: The output tensor of the model.

        Raises:
            Exception: If `x` is None, indicating an invalid input.
        """
        if x is not None:
            residual = self.conv_block(x)
            output = residual + self.repetitive_block(residual)

            return self.output_block(output)
        else:
            raise Exception("Please provide a valid input".capitalize())

    @staticmethod
    def total_params(model):
        """
        Calculates the total number of parameters in the model.

        Parameters:
            model (DnCNN): The model instance for which to calculate parameters.

        Returns:
            int: The total number of parameters in the model.

        Raises:
            Exception: If `model` is None, indicating a missing model instance.
        """
        if model is not None:
            return sum(p.numel() for p in model.parameters())
        else:
            raise Exception("Please provide a valid model".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Define the DnCNN model".title())

    parser.add_argument("--image_size", type=int, default=64, help="The image size")
    parser.add_argument("--in_channels", type=int, default=3, help="The input channels")
    parser.add_argument(
        "--out_channels", type=int, default=3, help="The output channels"
    )

    args = parser.parse_args()

    if args.image_size:
        model = DnCNN(
            image_size=args.image_size,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
        )

        print(DnCNN.total_params(model=model))

    else:
        raise Exception("Please provide a valid image size".capitalize())
