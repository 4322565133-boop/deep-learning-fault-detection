import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .unet_parts import DoubleConv


class MaxViTUNet(nn.Module):
    """
    MaxViT encoder (timm features_only) + U-Net-like CNN decoder.
    Robust to timm output format (NCHW/NHWC): auto-permute to NCHW.
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        backbone_name: str = "maxvit_tiny_tf_224",
        pretrained: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.backbone_name = backbone_name

        # try force NCHW, fallback if timm doesn't support output_fmt
        try:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
                output_fmt="NCHW",
            )
        except TypeError:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )

        self.chs = list(self.backbone.feature_info.channels())
        if len(self.chs) < 4:
            raise ValueError(f"{backbone_name} features_only must provide 4 scales, got {len(self.chs)}")

        self.dec3 = DoubleConv(self.chs[3] + self.chs[2], self.chs[2])
        self.dec2 = DoubleConv(self.chs[2] + self.chs[1], self.chs[1])
        self.dec1 = DoubleConv(self.chs[1] + self.chs[0], self.chs[0])

        self.dec0 = DoubleConv(self.chs[0], 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    @staticmethod
    def _ensure_nchw(feat: torch.Tensor, expected_c: int) -> torch.Tensor:
        # NCHW: [B,C,H,W], NHWC: [B,H,W,C]
        if feat.ndim == 4 and feat.shape[1] != expected_c and feat.shape[-1] == expected_c:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Most timm pretrained backbones expect 3-channel input
        x_in = x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x

        feats = self.backbone(x_in)
        if not isinstance(feats, (list, tuple)):
            feats = list(feats)
        feats = feats[:4]

        f0, f1, f2, f3 = [self._ensure_nchw(f, c) for f, c in zip(feats, self.chs)]

        d3 = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, f2], dim=1))

        d2 = F.interpolate(d3, size=f1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, f1], dim=1))

        d1 = F.interpolate(d2, size=f0.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, f0], dim=1))

        d0 = self.dec0(d1)
        d0 = F.interpolate(d0, size=x.shape[2:], mode="bilinear", align_corners=False)

        return self.outc(d0)
