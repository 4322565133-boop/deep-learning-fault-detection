import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DeiT3Seg(nn.Module):
    """
    DeiT3 / ViT encoder + simple segmentation head.
    - Use timm VisionTransformer forward_features to get tokens.
    - Reshape patch tokens to 2D feature map, then upsample to input resolution.

    Notes:
    - Input grayscale is repeated to 3-channel if needed.
    - Output is logits (no sigmoid).
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        backbone_name: str = "deit3_small_patch16_224",
        pretrained: bool = True,
        embed_dim: int = None,   # optional override
        head_dim: int = 256,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.backbone_name = backbone_name

        # Create ViT-like model (num_classes=0 disables classifier head in timm)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",   # keep tokens behavior
        )

        # infer embedding dim
        if embed_dim is None:
            # common timm ViT attribute names
            embed_dim = getattr(self.backbone, "embed_dim", None)
            if embed_dim is None:
                embed_dim = getattr(self.backbone, "num_features", None)
            if embed_dim is None:
                raise ValueError("Cannot infer embed_dim from backbone; pass embed_dim explicitly.")
        self.embed_dim = int(embed_dim)

        # segmentation decoder (lightweight)
        self.proj = ConvBNReLU(self.embed_dim, head_dim, k=1, p=0)
        self.refine = nn.Sequential(
            ConvBNReLU(head_dim, head_dim, k=3, p=1),
            ConvBNReLU(head_dim, head_dim, k=3, p=1),
        )
        self.outc = nn.Conv2d(head_dim, n_classes, kernel_size=1)

    def _tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, C] (patch tokens, without CLS)
        return: [B, C, H, W]
        """
        B, N, C = tokens.shape
        S = int(math.sqrt(N))
        if S * S != N:
            raise ValueError(f"Patch token count N={N} is not a perfect square; cannot reshape to 2D map.")
        x = tokens.transpose(1, 2).contiguous().view(B, C, S, S)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DeiT3 pretrained expects 3-channel
        x_in = x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x

        # timm ViT forward_features usually returns token sequence [B, 1+N, C]
        feats = self.backbone.forward_features(x_in)

        if feats.ndim != 3:
            # Some timm models may return [B, C] pooled feature; this is not suitable for segmentation.
            raise RuntimeError(f"{self.backbone_name} forward_features returned shape {tuple(feats.shape)}, expected [B, T, C].")

        # remove CLS token if present (common: token count = 1 + N)
        if feats.shape[1] > 1:
            patch_tokens = feats[:, 1:, :]
        else:
            raise RuntimeError("Token sequence has no patch tokens; cannot do segmentation.")

        fmap = self._tokens_to_map(patch_tokens)  # [B, C, S, S]
        fmap = self.proj(fmap)
        fmap = F.interpolate(fmap, size=x.shape[2:], mode="bilinear", align_corners=False)
        fmap = self.refine(fmap)
        logits = self.outc(fmap)
        return logits
