import sys
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models

class VGG16(nn.Module):
    """
    A custom VGG16-based model tailored for image processing tasks that may involve
    upsampling or feature transformation. This class adapts the original VGG16 model
    to allow modifications in the output layer to cater to different numbers of output channels
    and to upscale images to a specific size.

    Attributes:
        out_channels (int): The number of channels in the model's output.
        image_size (int): The target size of the output images.
        scale_factor (int): Factor by which images are upscaled.
        repetitive (int): Number of times the upsampling is applied.
        in_channels (int): Number of input channels to the first convolutional layer in the output block.
        config (list): Configuration for the convolutional layers in the output block, specifying in/out channels.
        model (torch.nn.Module): Feature extractor part of the VGG16 model.
    """

    def __init__(self, out_channels=3, image_size=128):
        """
        Initializes the VGG16 model with specified output channels and target image size.

        Parameters:
            out_channels (int): The desired number of channels for the output image.
            image_size (int): The target size for the output image.
        """
        super(VGG16, self).__init__()

        self.out_channels = out_channels
        self.image_size = image_size

        self.scale_factor = 2  # Factor by which images will be upscaled.
        self.repetitive = 5  # Number of upsampling iterations.
        self.in_channels = 512  # Starting number of channels for the output block.

        # Configuration for the output block's convolutional layers.
        self.config = [
            (self.in_channels, self.in_channels // 2),
            (self.in_channels // 2, self.in_channels // 4),
            (self.in_channels // 4, self.in_channels // 8),
            (self.in_channels // 8, self.out_channels),
        ]

        # Using the VGG16 model as a feature extractor.
        self.model = models.vgg16(pretrained=True).features
        for params in self.model.parameters():
            params.requires_grad = False  # Freeze the model to prevent updates.

        self.up_scaling = self.up_sampling_block()  # Defines the upsampling method.
        self.out = self.out_block(self.config)  # Constructs the output block.

    def up_sampling_block(self):
        """
        Creates an upsampling layer to increase the resolution of images.

        Returns:
            A bilinear upsampling layer.
        """
        return nn.Upsample(scale_factor=self.scale_factor, mode="bilinear", align_corners=False)

    def out_block(self, config):
        """
        Constructs the output block of the model with configurable convolutional layers.

        Parameters:
            config (list of tuple): Configuration specifying the in_channels and out_channels for each convolutional layer.

        Returns:
            A sequential container of convolutional and ReLU layers as defined in the config.
        """
        layers = OrderedDict()

        # Adding convolutional and ReLU layers based on the provided configuration.
        for index, (in_channels, out_channels) in enumerate(config[:-1]):
            layers[f"conv{index}"] = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
            layers[f"relu{index}"] = nn.ReLU(inplace=True)

        # Adding the final convolutional layer without ReLU to allow for different types of output (e.g., images).
        in_channels, out_channels = config[-1]
        layers[f"conv{len(config) - 1}"] = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)

        return nn.Sequential(layers)

    def forward(self, x):
        """
        Defines the forward pass of the VGG16 model.

        Parameters:
            x (torch.Tensor): Input tensor representing a batch of images.

        Returns:
            A torch.Tensor: The output tensor after processing by the model.
        """
        x = self.model(x)  # Apply the VGG16 feature extractor.

        # Upsample the output repeatedly to increase the image resolution.
        for _ in range(self.repetitive):
            x = self.up_scaling(x)

        return self.out(x)  # Apply the output block to get the final output.

if __name__ == "__main__":
    model = VGG16(out_channels=3)
    print(model(torch.randn(64, 3, 64, 64)).shape)  # Demonstrating usage of the model.
