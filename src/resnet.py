import sys
import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    """
    A custom ResNet-based model for image transformation tasks. 

    This class modifies the pre-trained ResNet-50 model by removing the final fully connected layers
    and adding a custom output block for image upscaling or feature transformation. The main purpose
    is to take an input image and apply a series of transformations to modify its features or dimensions.

    Attributes:
        out_channels (int): The number of channels in the output image.
        model (torch.nn.Module): The modified ResNet-50 model without the final layers.
    """

    def __init__(self, out_channels=3):
        """
        Initializes the ResNet model with a specified number of output channels.

        Parameters:
            out_channels (int): The number of output channels desired for the final output image.
        """
        super(ResNet, self).__init__()
        self.out_channels = out_channels

        self.model = models.resnet50(pretrained=True)

        # Freeze the parameters of the pre-trained model to avoid updating them during training.
        for params in self.model.parameters():
            params.requires_grad = False

        # Remove the last two layers (average pooling and fully connected layer) from the pre-trained ResNet-50 model.
        self.model = nn.Sequential(*list(self.model.children())[:-2])

    def upsample(self):
        """
        Creates an upsampling layer to increase the resolution of an image.

        Returns:
            An upsampling layer with bilinear interpolation.
        """
        return nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def out_block(self):
        """
        Constructs the final output block of the model.

        This block consists of convolutional layers designed to adjust the dimensions and
        features of the input image to produce the final output.

        Returns:
            A sequential layer comprising convolutional, batch normalization, and ReLU layers
            that process the image to the desired output format.
        """
        return nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                512, self.out_channels, kernel_size=3, padding=1, stride=1, bias=False
            ),
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
            x (torch.Tensor): The input tensor containing the images to be transformed.

        Returns:
            A torch.Tensor: The transformed images with the specified number of output channels.
        """
        x = self.model(x)  # Apply the modified ResNet-50 model.

        # Upsample the output from the model to the desired size.
        for _ in range(5):
            x = self.upsample()(x)

        # Pass the upsampled output through the final output block.
        return self.out_block()(x)

if __name__ == "__main__":
    model = ResNet()
    print(model(torch.randn(1, 3, 64, 64)).shape)  # Example use of the model to print the shape of the output tensor.
