import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .unet_parts import DoubleConv


class SwinUNet(nn.Module):
    """
    Swin backbone (timm features_only) + U-Net like decoder.
    Fix: timm backbone may output NHWC (channels_last) features on some versions/configs.
         We convert each feature map to NCHW before feeding decoder.
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        backbone_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.backbone_name = backbone_name

        # Some timm versions support output_fmt="NCHW"; older ones don't.
        # We try it first, fallback if not supported.
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

        # channels for each feature level
        self.chs = list(self.backbone.feature_info.channels())  # e.g. [96, 192, 384, 768] for swin_tiny

        # Decoder: (f3 up -> cat f2) -> dec3 -> (up -> cat f1) -> dec2 -> (up -> cat f0) -> dec1
        self.dec3 = DoubleConv(self.chs[3] + self.chs[2], self.chs[2])
        self.dec2 = DoubleConv(self.chs[2] + self.chs[1], self.chs[1])
        self.dec1 = DoubleConv(self.chs[1] + self.chs[0], self.chs[0])

        self.dec0 = DoubleConv(self.chs[0], 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def _ensure_nchw(self, feat: torch.Tensor, expected_c: int) -> torch.Tensor:
        """
        Convert feature map to NCHW if it's NHWC.
        NCHW: [B, C, H, W]
        NHWC: [B, H, W, C]
        """
        if feat.ndim != 4:
            return feat

        # If channel dimension isn't expected_c but last dimension is expected_c -> assume NHWC
        if feat.shape[1] != expected_c and feat.shape[-1] == expected_c:
            feat = feat.permute(0, 3, 1, 2).contiguous()

        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swin pretrained expects 3-channel input. If input is 1-channel, repeat to 3.
        x_in = x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x

        feats = self.backbone(x_in)
        # timm returns list/tuple of feature maps
        if not isinstance(feats, (list, tuple)):
            feats = list(feats)

        feats = feats[:4]
        f0, f1, f2, f3 = [
            self._ensure_nchw(f, c) for f, c in zip(feats, self.chs)
        ]

        # U-Net style decode
        d3 = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, f2], dim=1))

        d2 = F.interpolate(d3, size=f1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, f1], dim=1))

        d1 = F.interpolate(d2, size=f0.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, f0], dim=1))

        d0 = self.dec0(d1)
        d0 = F.interpolate(d0, size=x.shape[2:], mode="bilinear", align_corners=False)

        # return logits (sigmoid is applied in loss/metrics)
        return self.outc(d0)
