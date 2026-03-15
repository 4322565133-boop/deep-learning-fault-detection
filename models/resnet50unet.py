import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from .unet_parts import DoubleConv

class ResNet50_UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.enc0 = nn.Sequential(self.conv1, self.bn1, self.relu)

        self.dec4 = DoubleConv(2048 + 1024, 512)
        self.dec3 = DoubleConv(512 + 512, 256)
        self.dec2 = DoubleConv(256 + 256, 128)
        self.dec1 = DoubleConv(128 + 64, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x_in = x.repeat(1,3,1,1) if x.size(1)==1 else x
        x0 = self.enc0(x_in)                  # 64
        x1 = self.layer1(self.maxpool(x0))    # 256
        x2 = self.layer2(x1)                  # 512
        x3 = self.layer3(x2)                  # 1024
        x4 = self.layer4(x3)                  # 2048

        d4 = F.interpolate(x4, size=x3.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, x3], dim=1))
        d3 = F.interpolate(d4, size=x2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, x2], dim=1))
        d2 = F.interpolate(d3, size=x1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, x1], dim=1))
        d1 = F.interpolate(d2, size=x0.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, x0], dim=1))

        d1 = F.interpolate(d1, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.outc(d1)
