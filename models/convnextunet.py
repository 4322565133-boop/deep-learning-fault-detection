import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .unet_parts import DoubleConv


class ConvNeXtUNet(nn.Module):
    """
    ConvNeXt encoder (timm features_only) + U-Net-like CNN decoder.
    Fix: force features/logits to be contiguous to avoid `.view()` crash.
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        backbone_name: str = "convnext_tiny",
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

        # Decoder (U-Net style)
        self.dec3 = DoubleConv(self.chs[3] + self.chs[2], self.chs[2])
        self.dec2 = DoubleConv(self.chs[2] + self.chs[1], self.chs[1])
        self.dec1 = DoubleConv(self.chs[1] + self.chs[0], self.chs[0])

        self.dec0 = DoubleConv(self.chs[0], 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    @staticmethod
    def _ensure_nchw_contig(feat: torch.Tensor, expected_c: int) -> torch.Tensor:
        """
        - If timm returns NHWC, permute to NCHW
        - Always return contiguous tensor to avoid `.view()` failure downstream
        """
        # NHWC: [B,H,W,C]
        if feat.ndim == 4 and feat.shape[1] != expected_c and feat.shape[-1] == expected_c:
            feat = feat.permute(0, 3, 1, 2)
        return feat.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ConvNeXt pretrained expects 3-channel input
        x_in = x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x

        feats = self.backbone(x_in)
        if not isinstance(feats, (list, tuple)):
            feats = list(feats)
        feats = feats[:4]

        f0, f1, f2, f3 = [self._ensure_nchw_contig(f, c) for f, c in zip(feats, self.chs)]

        d3 = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, f2], dim=1).contiguous())

        d2 = F.interpolate(d3, size=f1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, f1], dim=1).contiguous())

        d1 = F.interpolate(d2, size=f0.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, f0], dim=1).contiguous())

        d0 = self.dec0(d1)
        d0 = F.interpolate(d0, size=x.shape[2:], mode="bilinear", align_corners=False)

        logits = self.outc(d0)
        return logits.contiguous()
