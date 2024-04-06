import sys
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models


class VGG16(nn.Module):
    def __init__(self, out_channels=3, image_size=128):
        super(VGG16, self).__init__()

        self.out_channels = out_channels
        self.image_size = image_size

        self.scale_factor = 2
        self.repetitive = 5
        self.in_channels = 512

        self.config = [
            (self.in_channels, self.in_channels // 2),
            (self.in_channels // 2, self.in_channels // 4),
            (self.in_channels // 4, self.in_channels // 8),
            (self.in_channels // 8, self.out_channels),
        ]

        self.model = models.vgg16(pretrained=True)
        self.model = self.model.features

        for params in self.model.parameters():
            params.requires_grad = False

        self.up_scaling = self.up_sampling_block()
        self.out = self.out_block(self.config)

    def up_sampling_block(self):
        return nn.Upsample(
            scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )

    def out_block(self, config):
        layers = OrderedDict()

        if config is not None:
            for index, (in_channels, out_channels) in enumerate(config[:-1]):
                layers["conv{}".format(index)] = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                )
                layers["relu{}".format(index)] = nn.ReLU(inplace=True)

            in_channels, out_channels = config[-1]
            layers["conv{}".format(len(config) - 1)] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            )

            return nn.Sequential(layers)
        else:
            raise Exception("config is None".capitalize())

    def forward(self, x):
        x = self.model(x)

        for _ in range(self.repetitive):
            x = self.up_scaling(x)

        return self.out(x)


if __name__ == "__main__":
    model = VGG16(out_channels=3)
    print(model(torch.randn(64, 3, 64, 64)).shape)
