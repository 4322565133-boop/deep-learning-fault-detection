import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, rates=(1, 6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                    )
                )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode="bilinear", align_corners=False)
        res.append(gp)
        x = torch.cat(res, dim=1)
        return self.project(x)


class TransDeepLab(nn.Module):
    """
    Swin encoder (timm features_only) + DeepLab-like ASPP + shallow low-level skip decoder.

    Fix:
    - timm Swin may output NHWC features on some versions/configs.
      We ensure each feature map is converted to NCHW before any Conv2d / cat.
    """

    def __init__(
        self,
        n_channels=1,
        n_classes=1,
        backbone_name="swin_tiny_patch4_window7_224",
        pretrained=True,
        low_level_ch=48,
    ):
        super().__init__()

        # Try to force NCHW output if timm supports it; otherwise fallback and permute manually in forward.
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

        self.chs = list(self.backbone.feature_info.channels())  # e.g. [96,192,384,768] for swin_tiny

        self.aspp = ASPP(self.chs[3], out_ch=256)
        self.low_proj = nn.Sequential(
            nn.Conv2d(self.chs[0], low_level_ch, 1, bias=False),
            nn.BatchNorm2d(low_level_ch),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + low_level_ch, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.outc = nn.Conv2d(256, n_classes, 1)

    @staticmethod
    def _ensure_nchw(feat: torch.Tensor, expected_c: int) -> torch.Tensor:
        """
        Convert NHWC -> NCHW when needed.
        NCHW: [B, C, H, W]
        NHWC: [B, H, W, C]
        """
        if feat.ndim != 4:
            return feat

        # if channel dim isn't expected but last dim is expected -> assume NHWC
        if feat.shape[1] != expected_c and feat.shape[-1] == expected_c:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x):
        # Swin pretrained expects 3 channels; repeat grayscale to RGB-like
        x_in = x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x

        feats = self.backbone(x_in)
        if not isinstance(feats, (list, tuple)):
            feats = list(feats)

        feats = feats[:4]
        f0, f1, f2, f3 = [self._ensure_nchw(f, c) for f, c in zip(feats, self.chs)]

        # ASPP on deepest feature
        x_aspp = self.aspp(f3)
        x_aspp = F.interpolate(x_aspp, size=f0.shape[2:], mode="bilinear", align_corners=False)

        # low-level feature projection
        low = self.low_proj(f0)

        # decoder
        x_cat = torch.cat([x_aspp, low], dim=1)
        x_dec = self.decoder(x_cat)

        # up to input resolution
        x_dec = F.interpolate(x_dec, size=x.shape[2:], mode="bilinear", align_corners=False)

        # return logits (sigmoid handled in loss/metrics)
        return self.outc(x_dec)
