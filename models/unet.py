import torch.nn as nn
from .unet_parts import DoubleConv, Down, Up, OutConv

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, base_c=64):
        super().__init__()
        self.inc = DoubleConv(n_channels, base_c)
        self.down1 = Down(base_c, base_c*2)
        self.down2 = Down(base_c*2, base_c*4)
        self.down3 = Down(base_c*4, base_c*8)
        self.down4 = Down(base_c*8, base_c*8)
        self.up1 = Up(base_c*16, base_c*4, bilinear)
        self.up2 = Up(base_c*8, base_c*2, bilinear)
        self.up3 = Up(base_c*4, base_c, bilinear)
        self.up4 = Up(base_c*2, base_c, bilinear)
        self.outc = OutConv(base_c, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
