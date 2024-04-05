import sys
import os
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params


class DnCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, image_size=64):
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
        if x is not None:
            residual = self.conv_block(x)
            output = residual + self.repetitive_block(residual)

            return self.output_block(output)
        else:
            raise Exception("Please provide a valid input".capitalize())

    @staticmethod
    def total_params(model):
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
