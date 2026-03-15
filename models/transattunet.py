import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import DoubleConv, Down, Up, OutConv
from .unet_parts_att_transformer import PositionEmbeddingLearned, PAM_Module, ScaledDotProductAttention
from .unet_parts_att_multiscale import MultiConv

class TransAttUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, base_c=64, att_heads=8):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, base_c)
        self.down1 = Down(base_c, base_c*2)
        self.down2 = Down(base_c*2, base_c*4)
        self.down3 = Down(base_c*4, base_c*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c*8, (base_c*16)//factor)

        self.up1 = Up((base_c*16)//factor + base_c*8, base_c*8//factor, bilinear)
        self.up2 = Up((base_c*8//factor) + base_c*4, base_c*4//factor, bilinear)
        self.up3 = Up((base_c*4//factor) + base_c*2, base_c*2//factor, bilinear)
        self.up4 = Up((base_c*2//factor) + base_c, base_c, bilinear)
        self.outc = OutConv(base_c, n_classes)

        bottleneck_c = (base_c*16)//factor
        self.pos = PositionEmbeddingLearned(bottleneck_c//2)
        self.pam = PAM_Module(bottleneck_c)
        self.sdpa = ScaledDotProductAttention(bottleneck_c, num_heads=att_heads)

        self.fuse1 = MultiConv(bottleneck_c + base_c*8//factor, base_c*8//factor)
        self.fuse2 = MultiConv((base_c*8//factor) + base_c*4//factor, base_c*4//factor)
        self.fuse3 = MultiConv((base_c*4//factor) + base_c*2//factor, base_c*2//factor)
        self.fuse4 = MultiConv((base_c*2//factor) + base_c, base_c)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5_pam = self.pam(x5)
        x5_pos = self.pos(x5)
        x5 = x5 + x5_pos
        x5_sdpa = self.sdpa(x5)
        x5 = x5_sdpa + x5_pam

        x6 = self.up1(x5, x4)
        x6 = self.fuse1(torch.cat([x6, F.interpolate(x5, size=x6.shape[2:], mode="bilinear", align_corners=False)], dim=1))

        x7 = self.up2(x6, x3)
        x7 = self.fuse2(torch.cat([x7, F.interpolate(x6, size=x7.shape[2:], mode="bilinear", align_corners=False)], dim=1))

        x8 = self.up3(x7, x2)
        x8 = self.fuse3(torch.cat([x8, F.interpolate(x7, size=x8.shape[2:], mode="bilinear", align_corners=False)], dim=1))

        x9 = self.up4(x8, x1)
        x9 = self.fuse4(torch.cat([x9, F.interpolate(x8, size=x9.shape[2:], mode="bilinear", align_corners=False)], dim=1))

        return self.outc(x9)
