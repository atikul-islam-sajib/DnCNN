import sys
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, out_channels=3):
        super(ResNet, self).__init__()
        self.out_channels = out_channels

        self.model = models.resnet50(pretrained=True)

        for params in self.model.parameters():
            params.requires_grad = False

        self.model = nn.Sequential(*list(self.model.children())[:-2])

    def upsample(self):
        return nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def out_block(self):
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
        x = self.model(x)

        for _ in range(5):
            x = self.upsample()(x)

        return self.out_block()(x)


if __name__ == "__main__":
    model = ResNet()
    print(model(torch.randn(1, 3, 64, 64)).shape)
