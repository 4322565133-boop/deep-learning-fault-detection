import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from .unet_parts import DoubleConv

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )
    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + h
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h
        return x

class TransUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, pretrained=True, trans_depth=2, trans_heads=8, embed_dim=1024):
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

        self.proj_in = nn.Conv2d(2048, embed_dim, 1)
        self.proj_out = nn.Conv2d(embed_dim, 2048, 1)
        self.pos_embed = None
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads=trans_heads) for _ in range(trans_depth)])

        self.dec4 = DoubleConv(2048 + 1024, 512)
        self.dec3 = DoubleConv(512 + 512, 256)
        self.dec2 = DoubleConv(256 + 256, 128)
        self.dec1 = DoubleConv(128 + 64, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def _ensure_pos(self, N, C, device):
        if (self.pos_embed is None) or (self.pos_embed.shape[1] != N) or (self.pos_embed.shape[2] != C):
            self.pos_embed = nn.Parameter(torch.zeros(1, N, C, device=device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x_in = x.repeat(1,3,1,1) if x.size(1)==1 else x
        x0 = self.enc0(x_in)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        b = self.proj_in(x4)
        B,C,H,W = b.shape
        tokens = b.flatten(2).transpose(1,2)
        self._ensure_pos(tokens.shape[1], tokens.shape[2], tokens.device)
        tokens = tokens + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        b2 = tokens.transpose(1,2).reshape(B,C,H,W)
        x4t = self.proj_out(b2)

        d4 = F.interpolate(x4t, size=x3.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, x3], dim=1))
        d3 = F.interpolate(d4, size=x2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, x2], dim=1))
        d2 = F.interpolate(d3, size=x1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, x1], dim=1))
        d1 = F.interpolate(d2, size=x0.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, x0], dim=1))
        d1 = F.interpolate(d1, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.outc(d1)
